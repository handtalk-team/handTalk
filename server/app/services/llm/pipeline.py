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
from typing import List, Optional

from app.core.config import get_settings
from app.models.schemas.ws_messages import AvatarCommand, LLMResponse

logger = logging.getLogger(__name__)
settings = get_settings()

# ─────────────────────── 시스템 프롬프트 ─────────────────────────

_SYSTEM_PROMPT_BASE = """You are a Korean Sign Language (KSL) tutor AI. IMPORTANT RULES:
1. ALWAYS respond in Korean only. Never use English, Japanese, Chinese, or any other language.
2. Keep responses to 1-2 sentences maximum.
3. Be warm and encouraging toward the learner.
4. If the user made a signing error, start your response with "(교정)".

Context: The user is practicing sign language for hospital/medical settings.
Scenario: {scenario}
Detail: {scenario_detail}

Remember: Korean responses ONLY. Short and clear."""

_SCENARIO_DETAILS = {
    "free_talk": "자유 주제로 대화합니다. 사용자가 말하는 내용에 자연스럽게 반응하세요.",
    "greetings": "인사말과 자기소개 위주의 학습입니다. 안녕하세요, 감사합니다, 이름 등을 연습합니다.",
    "hospital":  "병원 환경에서 필요한 수어를 학습합니다. 아프다, 약, 의사, 간호사, 도와주세요 등을 집중합니다.",
    "numbers":   "숫자와 날짜 표현을 학습합니다.",
    "emotions":  "감정 표현 수어를 학습합니다. 좋아요, 슬프다, 화나다, 무섭다 등을 연습합니다.",
}

# 내장 응답 (LLM이 없을 때 fallback) — 추가 단어는 직접 넣으세요
_MOCK_RESPONSES = {
    "안녕하세요": "안녕하세요! 수어 표현이 정확해요. 계속 연습해봐요!",
    "좋아요":     "좋아요! 엄지 표현이 잘 됐어요.",
    "브이":       "브이! 두 손가락 표현이 깔끔해요.",
}


class LLMPipeline:
    """세션 하나당 인스턴스 하나. 프로바이더를 자동으로 선택합니다."""

    MAX_HISTORY = 20
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
        self._error_context: List[str] = []
        self._client = None
        self._provider = settings.LLM_PROVIDER

    async def start(self) -> None:
        """프로바이더에 맞는 클라이언트를 초기화합니다."""
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
            logger.info("LLM 프로바이더: %s", self._provider)

    async def chat(
        self,
        user_sign_text: str,
        error_hints: Optional[List[str]] = None,
    ) -> LLMResponse:
        if error_hints:
            self._error_context.extend(error_hints)
            self._error_context = self._error_context[-5:]

        user_content = user_sign_text
        if error_hints:
            user_content += f"\n[인식된 오류: {', '.join(error_hints)}]"

        self._history.append({"role": "user", "content": user_content})

        if self._client is None:
            return self._mock_response(user_sign_text)

        system_prompt = self._build_system_prompt()
        reply_text = ""
        tokens = 0

        try:
            if self._provider == "claude":
                reply_text, tokens = await self._chat_claude(system_prompt)
            else:
                # Ollama, Groq 모두 OpenAI 호환 API
                reply_text, tokens = await self._chat_openai_compat(system_prompt)
        except Exception as e:
            logger.error("LLM 호출 실패 (%s): %s", self._provider, e)
            return self._mock_response(user_sign_text)

        if not reply_text:
            return self._mock_response(user_sign_text)

        self._history.append({"role": "assistant", "content": reply_text})

        if self._redis:
            await self._redis.setex(
                f"history:{self._session_id}",
                self.CACHE_TTL_S,
                json.dumps(self._history[-self.MAX_HISTORY:]),
            )

        from app.services.avatar.controller import AvatarController
        return LLMResponse(
            text=reply_text,
            avatar_commands=AvatarController.text_to_commands(reply_text),
            tokens_used=tokens,
        )

    async def stop(self) -> None:
        if self._client and hasattr(self._client, "close"):
            await self._client.close()

    # ─────────────── 프로바이더별 초기화 ─────────────────────────

    def _init_ollama(self):
        """
        Ollama 로컬 서버에 연결합니다.
        사전 준비:
          brew install ollama        # macOS
          ollama run llama3.2        # 첫 실행 시 모델 다운로드 (~2GB)
        """
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                base_url=settings.OLLAMA_BASE_URL,
                api_key="ollama",          # Ollama는 키 불필요 — 아무 문자열
            )
            logger.info("Ollama 클라이언트 초기화 완료 (base_url=%s, model=%s)",
                        settings.OLLAMA_BASE_URL, settings.OLLAMA_MODEL)
            return client
        except ImportError:
            logger.error("openai 패키지가 없습니다. pip install openai")
            return None

    def _init_groq(self):
        """
        Groq 무료 클라우드 API를 사용합니다.
        사전 준비:
          1. https://console.groq.com 에서 무료 계정 생성
          2. API 키 발급 (gsk_...)
          3. .env 에 GROQ_API_KEY=gsk_... 추가
        """
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
        """Ollama & Groq 공통 호출 (OpenAI SDK)."""
        model = (
            settings.GROQ_MODEL if self._provider == "groq"
            else settings.OLLAMA_MODEL
        )
        messages = [{"role": "system", "content": system_prompt}]
        messages += self._history[-self.MAX_HISTORY:]

        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=0.7,
        )
        text = response.choices[0].message.content.strip()
        tokens = (
            response.usage.total_tokens
            if response.usage else 0
        )
        return text, tokens

    async def _chat_claude(self, system_prompt: str):
        """Anthropic Claude 호출."""
        response = await self._client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=settings.LLM_MAX_TOKENS,
            system=system_prompt,
            messages=self._history[-self.MAX_HISTORY:],
        )
        text = response.content[0].text.strip()
        tokens = (
            response.usage.input_tokens + response.usage.output_tokens
            if hasattr(response, "usage") else 0
        )
        return text, tokens

    # ─────────────── 공통 유틸 ────────────────────────────────────

    def _build_system_prompt(self) -> str:
        detail = _SCENARIO_DETAILS.get(self._scenario, _SCENARIO_DETAILS["hospital"])
        prompt = _SYSTEM_PROMPT_BASE.format(
            scenario=self._scenario,
            scenario_detail=detail,
        )
        if self._error_context:
            prompt += (
                f"\n\n[최근 오류 패턴: {', '.join(set(self._error_context))}]\n"
                "위 오류를 고려해 맞춤형 피드백을 자연스럽게 포함하세요."
            )
        return prompt

    def _mock_response(self, user_text: str) -> LLMResponse:
        """LLM 없을 때 내장 응답 사용."""
        text = _MOCK_RESPONSES.get(
            user_text,
            f"'{user_text}' 수어를 잘 표현했어요. 계속 연습해보세요!",
        )
        from app.services.avatar.controller import AvatarController
        return LLMResponse(
            text=text,
            avatar_commands=AvatarController.text_to_commands(text),
            tokens_used=0,
        )
