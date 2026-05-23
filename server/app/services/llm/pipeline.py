"""
LLM Conversation Pipeline
==========================

지원 프로바이더 (LLM_PROVIDER 환경변수로 선택)
-----------------------------------------------
  ollama  : 완전 무료 로컬 실행 (기본값)
  groq    : 무료 클라우드 API
  claude  : Anthropic Claude (유료)

응답 종류 (LLMResponse.kind)
-----------------------------
  prompt   : 새 단어 안내 — 고정 문구, 항상 동일
  correct  : 정답 칭찬 + 다음 단어 안내
  feedback : 오답 교정 — LLM이 생성, 매번 새로운 문장
"""

from __future__ import annotations

import logging
import os
import random
from typing import List, Optional

from app.core.config import get_settings
from app.models.schemas.ws_messages import LLMResponse

logger = logging.getLogger(__name__)
settings = get_settings()

# ─────────────────────── 시나리오 스크립트 ─────────────────────────
# 항상 동일하게 출력되는 고정 안내 문구
_SCENARIO_SCRIPTS: dict[str, list[dict]] = {
    "hospital": [
        {"word": "안녕하세요",  "prompt": "병원에 입장했습니다. 접수처 직원에게 '안녕하세요'를 해볼까요?"},
        {"word": "아프다",     "prompt": "진료실에 들어왔어요. 의사에게 어디가 '아프다'고 표현해볼까요?"},
        {"word": "병원",      "prompt": "친구에게 '병원'이 어디냐고 물어볼까요?"},
        {"word": "도와주세요",  "prompt": "응급 상황이에요! '도와주세요'를 크게 표현해볼까요?"},
        {"word": "약",        "prompt": "약국에 들렀어요. 약사에게 '약'을 달라고 해볼까요?"},
        {"word": "의사",      "prompt": "담당 '의사' 선생님을 찾고 있어요. 표현해볼까요?"},
        {"word": "간호사",     "prompt": "옆에 계신 '간호사' 선생님을 불러볼까요?"},
    ],
    "greetings": [
        {"word": "안녕하세요",  "prompt": "처음 만난 분이에요. '안녕하세요'를 표현해볼까요?"},
        {"word": "감사합니다",  "prompt": "친절하게 도와주셨어요. '감사합니다'를 표현해볼까요?"},
        {"word": "반갑습니다",  "prompt": "오랜만에 만난 친구예요. '반갑습니다'를 표현해볼까요?"},
    ],
    "emotions": [
        {"word": "좋아요",    "prompt": "오늘 기분이 어때요? '좋아요'를 표현해볼까요?"},
        {"word": "슬프다",    "prompt": "슬픈 일이 있었어요. '슬프다'를 표현해볼까요?"},
        {"word": "화나다",    "prompt": "화가 났어요. '화나다'를 표현해볼까요?"},
        {"word": "무섭다",    "prompt": "무서운 일이 생겼어요. '무섭다'를 표현해볼까요?"},
    ],
    "free_talk": [],
    "numbers":   [],
}

# 정답 칭찬 문구 — LLM 없이 랜덤 선택
_PRAISE_PHRASES = [
    "정확해요!",
    "잘하셨어요!",
    "훌륭합니다!",
    "완벽해요!",
    "아주 잘 표현하셨어요!",
    "멋지게 해내셨어요!",
    "훌륭한 동작이에요!",
    "완벽하게 표현하셨어요!",
]

# 피드백 LLM 시스템 프롬프트
_FEEDBACK_SYSTEM = """당신은 한국 수어(KSL) 전문 교정 튜터입니다.

철칙:
1. 반드시 한국어만 사용하세요. 영어, 중국어 등 다른 언어 절대 금지.
2. 1~2문장으로만 답하세요.
3. 손 모양, 위치, 움직임 방향을 구체적으로 안내하세요.
4. 따뜻하고 격려하는 어투로 말하세요.
5. "다시 해보세요" 같은 말은 하지 마세요. 교정 내용만 말하세요."""


class LLMPipeline:
    """세션 하나당 인스턴스 하나."""

    CACHE_TTL_S = 3600

    def __init__(
        self,
        session_id: str,
        scenario: str = "hospital",
        redis=None,
    ) -> None:
        self._session_id = session_id
        self._scenario = scenario
        self._redis = redis
        self._client = None
        self._provider = settings.LLM_PROVIDER

        # [{"word": str, "prompt": str}]
        self._steps: list[dict] = []
        self._step_index: int = 0

    async def start(self) -> None:
        vocab_list = self._load_vocab_list()
        self._steps = self._build_steps(vocab_list)

        if self._provider == "ollama":
            self._client = self._init_ollama()
        elif self._provider == "groq":
            self._client = self._init_groq()
        elif self._provider == "claude":
            self._client = self._init_claude()
        else:
            logger.warning("알 수 없는 LLM_PROVIDER='%s'. 피드백 없이 진행합니다.", self._provider)

        logger.info("LLM 프로바이더: %s, 학습 단계: %d개, 단계: %s",
                    self._provider, len(self._steps),
                    [s["word"] for s in self._steps])

    def get_opening_message(self) -> LLMResponse:
        """세션 시작 시 첫 단계 안내 — LLM 호출 없이 즉시 반환."""
        self._step_index = 0
        if self._steps:
            text = self._steps[0]["prompt"]
        else:
            text = "수어 연습을 시작해볼게요!"
        return self._make_response(text, kind="prompt")

    async def chat(
        self,
        user_sign_text: str,
        error_hints: Optional[List[str]] = None,
    ) -> LLMResponse:
        if not self._steps:
            return self._make_response("계속 연습해보세요!", kind="feedback")

        current = self._steps[self._step_index % len(self._steps)]
        target_word = current["word"]

        if user_sign_text == target_word:
            return self._handle_correct(target_word)
        else:
            return await self._handle_incorrect(target_word, user_sign_text)

    async def stop(self) -> None:
        if self._client and hasattr(self._client, "close"):
            await self._client.close()

    # ─────────────── 정답 / 오답 처리 ────────────────────────────

    def _handle_correct(self, word: str) -> LLMResponse:
        self._step_index += 1
        praise = random.choice(_PRAISE_PHRASES)

        if self._step_index >= len(self._steps):
            # 모든 단계 완료 → 처음부터
            self._step_index = 0
            next_prompt = self._steps[0]["prompt"]
            text = f"{praise} 모든 단어를 완료했어요! 정말 대단해요! 처음부터 다시 해볼게요. {next_prompt}"
        else:
            next_prompt = self._steps[self._step_index]["prompt"]
            text = f"{praise} {next_prompt}"

        return self._make_response(text, kind="correct")

    async def _handle_incorrect(self, target: str, recognized: str) -> LLMResponse:
        feedback = await self._get_feedback(target, recognized)
        text = f"{feedback} 다시 '{target}'를 표현해볼까요?"
        return self._make_response(text, kind="feedback")

    # ─────────────── LLM 피드백 호출 ─────────────────────────────

    async def _get_feedback(self, target: str, recognized: str) -> str:
        fallback = f"'{target}' 동작을 천천히 다시 한번 해보세요."

        if self._client is None:
            return fallback

        user_msg = (
            f"목표 수어: '{target}'. 인식된 동작: '{recognized}'. "
            f"'{target}' 동작을 올바르게 교정해주세요."
        )
        try:
            if self._provider == "claude":
                text, _ = await self._call_claude(_FEEDBACK_SYSTEM, user_msg)
            else:
                text, _ = await self._call_openai(_FEEDBACK_SYSTEM, user_msg)
        except Exception as e:
            logger.error("LLM 피드백 호출 실패: %s", e)
            return fallback

        return self._filter_korean(text, fallback)

    # ─────────────── 프로바이더별 초기화 ─────────────────────────

    def _init_ollama(self):
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(base_url=settings.OLLAMA_BASE_URL, api_key="ollama")
            logger.info("Ollama 초기화 완료 (model=%s)", settings.OLLAMA_MODEL)
            return client
        except ImportError:
            logger.error("openai 패키지가 없습니다. pip install openai")
            return None

    def _init_groq(self):
        if not settings.GROQ_API_KEY:
            logger.warning("GROQ_API_KEY가 없습니다.")
            return None
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=settings.GROQ_API_KEY,
            )
            logger.info("Groq 초기화 완료 (model=%s)", settings.GROQ_MODEL)
            return client
        except ImportError:
            logger.error("openai 패키지가 없습니다.")
            return None

    def _init_claude(self):
        if not settings.ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY가 없습니다.")
            return None
        try:
            import anthropic
            return anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        except ImportError:
            logger.error("anthropic 패키지가 없습니다.")
            return None

    # ─────────────── 단일 LLM 호출 (히스토리 없음) ───────────────

    async def _call_openai(self, system: str, user_msg: str):
        model = settings.GROQ_MODEL if self._provider == "groq" else settings.OLLAMA_MODEL
        resp = await self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=100,
            temperature=0.7,
        )
        text = resp.choices[0].message.content.strip()
        tokens = resp.usage.total_tokens if resp.usage else 0
        return text, tokens

    async def _call_claude(self, system: str, user_msg: str):
        resp = await self._client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=100,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = resp.content[0].text.strip()
        tokens = (resp.usage.input_tokens + resp.usage.output_tokens
                  if hasattr(resp, "usage") else 0)
        return text, tokens

    # ─────────────── 공통 유틸 ────────────────────────────────────

    def _load_vocab_list(self) -> list[str]:
        labels_path = os.path.join(
            os.path.dirname(settings.ONNX_MODEL_PATH), "labels.txt"
        )
        if os.path.exists(labels_path):
            with open(labels_path, encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        return ["안녕하세요", "아프다", "병원", "도와주세요"]

    def _build_steps(self, vocab_list: list[str]) -> list[dict]:
        """스크립트 단어 중 vocab에 있는 것만 선택. 나머지는 labels.txt 순서로 추가."""
        vocab_set = set(vocab_list)
        script = _SCENARIO_SCRIPTS.get(self._scenario, [])

        steps = [s for s in script if s["word"] in vocab_set]
        scripted_words = {s["word"] for s in steps}

        for word in vocab_list:
            if word not in scripted_words:
                steps.append({"word": word, "prompt": f"이번에는 '{word}'를 표현해볼까요?"})

        return steps

    def _make_response(self, text: str, kind: str, tokens: int = 0) -> LLMResponse:
        from app.services.avatar.controller import AvatarController
        return LLMResponse(
            text=text,
            avatar_commands=AvatarController.text_to_commands(text),
            tokens_used=tokens,
            kind=kind,
        )

    def _filter_korean(self, text: str, fallback: str) -> str:
        """일본어·중국어·영어가 섞이면 폴백 사용."""
        if not text:
            return fallback

        for ch in text:
            code = ord(ch)
            # 히라가나(3040-309F), 가타카나(30A0-30FF), CJK 한자(4E00-9FFF, 3400-4DBF)
            if (0x3040 <= code <= 0x30FF or
                    0x3400 <= code <= 0x4DBF or
                    0x4E00 <= code <= 0x9FFF):
                logger.warning("LLM 응답에 일본어/한자 포함, 폴백 사용: %r", text)
                return fallback

        # 영어 알파벳 비율 체크 (30% 초과 시 거부)
        ascii_alpha = sum(1 for c in text if ord(c) < 128 and c.isalpha())
        total_alpha = sum(1 for c in text if c.isalpha())
        if total_alpha > 0 and ascii_alpha / total_alpha > 0.3:
            logger.warning("LLM 응답에 영어 포함, 폴백 사용: %r", text)
            return fallback

        return text
