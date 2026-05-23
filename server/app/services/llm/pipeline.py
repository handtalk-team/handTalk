"""
LLM Conversation Pipeline
==========================

지원 프로바이더 (LLM_PROVIDER 환경변수로 선택)
-----------------------------------------------
  ollama  : 완전 무료 로컬 실행 (기본값)
              → brew install ollama && ollama run llama3.2
  groq    : 무료 클라우드 API (console.groq.com 에서 키 발급)
              → GROQ_API_KEY=gsk_... 설정
  claude  : Anthropic Claude (유료, ANTHROPIC_API_KEY 필요)

Ollama와 Groq는 OpenAI 호환 API를 제공하므로
openai Python SDK 하나로 두 프로바이더를 모두 처리합니다.

지연시간 목표: < 800ms (LLM 응답)
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional

from app.core.config import get_settings
from app.models.schemas.ws_messages import AvatarCommand, LLMResponse

logger = logging.getLogger(__name__)
settings = get_settings()

# ─────────────────────── 시스템 프롬프트 ─────────────────────────

# LLM은 오직 짧은 한국어 칭찬/반응만 생성. 단어 제안은 서버가 처리.
_SYSTEM_PROMPT_BASE = """당신은 한국 수어(KSL) 튜터입니다.

규칙 (반드시 지켜야 합니다):
1. 오직 한국어만 사용하세요. 영어, 중국어, 독일어 등 다른 언어 절대 금지.
2. 한 문장으로만 답하세요.
3. 사용자가 수어 동작을 했을 때 짧게 칭찬하세요.
4. 단어를 직접 추천하거나 제안하지 마세요. 칭찬만 하세요.

예시 응답:
- "잘하셨어요!"
- "동작이 정확해요!"
- "훌륭합니다!"
- "잘 표현하셨어요!"

시나리오: {scenario}"""


class LLMPipeline:
    """세션 하나당 인스턴스 하나. 프로바이더를 자동으로 선택합니다."""

    MAX_HISTORY = 10
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
        self._history: List[dict] = []
        self._client = None
        self._provider = settings.LLM_PROVIDER

        # 서버가 단어 순서를 직접 관리
        self._vocab: List[str] = []
        self._word_index: int = 0

    async def start(self) -> None:
        """프로바이더에 맞는 클라이언트를 초기화합니다."""
        self._vocab = self._load_vocab_list()

        if self._provider == "ollama":
            self._client = self._init_ollama()
        elif self._provider == "groq":
            self._client = self._init_groq()
        elif self._provider == "claude":
            self._client = self._init_claude()
        else:
            logger.warning("알 수 없는 LLM_PROVIDER='%s'. mock 응답을 사용합니다.", self._provider)
            self._client = None

        # Redis에서 대화 이력 복원
        if self._redis:
            raw = await self._redis.get(f"history:{self._session_id}")
            if raw:
                self._history = json.loads(raw)

        if self._client:
            logger.info("LLM 프로바이더: %s, 단어 목록: %s", self._provider, self._vocab)

    def get_opening_message(self) -> LLMResponse:
        """세션 시작 시 첫 번째 단어 안내 메시지를 반환합니다."""
        from app.services.avatar.controller import AvatarController
        if self._vocab:
            first_word = self._vocab[0]
            self._word_index = 0
            text = f"안녕하세요! 수어 연습을 시작해볼게요. 먼저 '{first_word}'를 표현해보세요!"
        else:
            text = "안녕하세요! 수어 연습을 시작해볼게요."
        return LLMResponse(
            text=text,
            avatar_commands=AvatarController.text_to_commands(text),
            tokens_used=0,
        )

    async def chat(
        self,
        user_sign_text: str,
        error_hints: Optional[List[str]] = None,
    ) -> LLMResponse:
        # 다음 연습 단어 결정 (서버가 직접 관리)
        next_word = self._get_next_word(user_sign_text)

        if self._client is None:
            return self._build_response(
                praise=f"'{user_sign_text}' 수어를 잘 표현했어요.",
                next_word=next_word,
            )

        # LLM에게는 짧은 칭찬 한 문장만 요청
        user_content = f"사용자가 '{user_sign_text}' 수어를 표현했습니다. 한 문장으로 칭찬해주세요."
        self._history.append({"role": "user", "content": user_content})

        system_prompt = _SYSTEM_PROMPT_BASE.format(scenario=self._scenario)
        praise = ""
        tokens = 0

        try:
            if self._provider == "claude":
                praise, tokens = await self._chat_claude(system_prompt)
            else:
                praise, tokens = await self._chat_openai_compat(system_prompt)
        except Exception as e:
            logger.error("LLM 호출 실패 (%s): %s", self._provider, e)

        if not praise:
            praise = f"'{user_sign_text}' 잘 표현하셨어요."

        # 한국어만 남기도록 간단 필터 (ASCII 비율이 높으면 폴백)
        praise = self._filter_korean(praise, user_sign_text)

        self._history.append({"role": "assistant", "content": praise})
        self._history = self._history[-self.MAX_HISTORY:]

        if self._redis:
            await self._redis.setex(
                f"history:{self._session_id}",
                self.CACHE_TTL_S,
                json.dumps(self._history),
            )

        return self._build_response(praise=praise, next_word=next_word, tokens=tokens)

    async def stop(self) -> None:
        if self._client and hasattr(self._client, "close"):
            await self._client.close()

    # ─────────────── 단어 순서 관리 ──────────────────────────────

    def _get_next_word(self, recognized: str) -> Optional[str]:
        """인식된 단어가 현재 목표 단어이면 다음 단어로 진행."""
        if not self._vocab:
            return None
        current = self._vocab[self._word_index % len(self._vocab)]
        if recognized == current:
            self._word_index += 1
        return self._vocab[self._word_index % len(self._vocab)]

    def _build_response(
        self,
        praise: str,
        next_word: Optional[str],
        tokens: int = 0,
    ) -> LLMResponse:
        from app.services.avatar.controller import AvatarController
        if next_word:
            text = f"{praise} 다음은 '{next_word}'를 표현해보세요!"
        else:
            text = praise
        return LLMResponse(
            text=text,
            avatar_commands=AvatarController.text_to_commands(text),
            tokens_used=tokens,
        )

    # ─────────────── 한국어 필터 ─────────────────────────────────

    def _filter_korean(self, text: str, fallback_word: str) -> str:
        """ASCII 문자 비율이 30% 초과이면 폴백 응답 사용."""
        if not text:
            return f"'{fallback_word}' 잘 표현하셨어요."
        ascii_count = sum(1 for c in text if ord(c) < 128 and c.isalpha())
        total_alpha = sum(1 for c in text if c.isalpha())
        if total_alpha > 0 and ascii_count / total_alpha > 0.3:
            logger.warning("LLM 응답에 비한국어 포함, 폴백 사용: %r", text)
            return f"'{fallback_word}' 잘 표현하셨어요."
        return text

    # ─────────────── 프로바이더별 초기화 ─────────────────────────

    def _init_ollama(self):
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                base_url=settings.OLLAMA_BASE_URL,
                api_key="ollama",
            )
            logger.info("Ollama 클라이언트 초기화 완료 (base_url=%s, model=%s)",
                        settings.OLLAMA_BASE_URL, settings.OLLAMA_MODEL)
            return client
        except ImportError:
            logger.error("openai 패키지가 없습니다. pip install openai")
            return None

    def _init_groq(self):
        if not settings.GROQ_API_KEY:
            logger.warning("GROQ_API_KEY가 없습니다. mock 응답으로 대체합니다.")
            return None
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=settings.GROQ_API_KEY,
            )
            logger.info("Groq 클라이언트 초기화 완료 (model=%s)", settings.GROQ_MODEL)
            return client
        except ImportError:
            logger.error("openai 패키지가 없습니다. pip install openai")
            return None

    def _init_claude(self):
        if not settings.ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY가 없습니다. mock 응답으로 대체합니다.")
            return None
        try:
            import anthropic
            return anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        except ImportError:
            logger.error("anthropic 패키지가 없습니다. pip install anthropic")
            return None

    # ─────────────── 프로바이더별 채팅 호출 ──────────────────────

    async def _chat_openai_compat(self, system_prompt: str):
        model = (
            settings.GROQ_MODEL if self._provider == "groq"
            else settings.OLLAMA_MODEL
        )
        messages = [{"role": "system", "content": system_prompt}]
        messages += self._history[-6:]

        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=80,
            temperature=0.7,
        )
        text = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens if response.usage else 0
        return text, tokens

    async def _chat_claude(self, system_prompt: str):
        response = await self._client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=80,
            system=system_prompt,
            messages=self._history[-6:],
        )
        text = response.content[0].text.strip()
        tokens = (
            response.usage.input_tokens + response.usage.output_tokens
            if hasattr(response, "usage") else 0
        )
        return text, tokens

    # ─────────────── 공통 유틸 ────────────────────────────────────

    def _load_vocab_list(self) -> List[str]:
        labels_path = os.path.join(
            os.path.dirname(settings.ONNX_MODEL_PATH), "labels.txt"
        )
        if os.path.exists(labels_path):
            with open(labels_path, encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        return ["안녕하세요", "아프다", "병원", "도와주세요"]
