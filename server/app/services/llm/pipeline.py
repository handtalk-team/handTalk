"""
LLM Conversation Pipeline
==========================

Responsibilities
----------------
1.  Maintains per-session conversation history (stored in Redis, capped at 20 turns).
2.  Builds a system prompt tailored to the current scenario and the user's
    recent error patterns (personalised tutoring).
3.  Calls Claude claude-sonnet-4-6 with prompt caching on the system prompt to cut
    latency and cost on repeated turns.
4.  Returns the assistant's Korean text and a list of AvatarCommand objects
    derived from the text.

Latency budget contribution : ~800 ms (LLM API round-trip).

Usage
-----
    pipeline = LLMPipeline(session_id="abc", scenario="greetings")
    await pipeline.start()
    response = await pipeline.chat("안녕하세요")
    print(response.text, response.avatar_commands)
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

import anthropic

from app.core.config import get_settings
from app.models.schemas.ws_messages import AvatarCommand, LLMResponse

logger = logging.getLogger(__name__)
settings = get_settings()

# ─────────────────────── system prompt ──────────────────────────

_SYSTEM_PROMPT_BASE = """당신은 한국수어(KSL) 전문 튜터 AI입니다.
사용자가 수어로 표현한 단어나 문장을 텍스트로 전달받으면, 자연스럽고 따뜻한 한국어로 대화를 이어갑니다.

[역할과 원칙]
- 수어 학습자를 격려하고, 틀린 동작은 부드럽게 교정합니다.
- 응답은 간결하게 1~3문장 이내로 유지합니다 (아바타 수어 표현 시간 제한).
- 수어로 표현 가능한 단어와 문장을 우선적으로 사용합니다.
- 의료 환경(병원) 상황에 맞는 어휘를 활용합니다.

[응답 형식]
- 자연스러운 한국어 대화 응답만 출력합니다.
- 영어나 특수기호는 사용하지 않습니다.
- 사용자가 틀린 수어를 했을 경우 "(교정)" 접두어로 피드백을 제공합니다.

[학습 시나리오: {scenario}]
{scenario_detail}"""

_SCENARIO_DETAILS = {
    "free_talk": "자유 주제로 대화합니다. 사용자가 말하는 내용에 자연스럽게 반응하세요.",
    "greetings": "인사말과 자기소개 위주의 학습입니다. 안녕하세요, 감사합니다, 이름 등을 연습합니다.",
    "hospital": "병원 환경에서 필요한 수어를 학습합니다. 아프다, 약, 의사, 간호사, 도와주세요 등을 집중합니다.",
    "numbers": "숫자와 날짜 표현을 학습합니다.",
    "emotions": "감정 표현 수어를 학습합니다. 좋아요, 슬프다, 화나다, 무섭다 등을 연습합니다.",
}


class LLMPipeline:
    """One instance per WebSocket session."""

    MAX_HISTORY = 20    # turns kept in Redis
    CACHE_TTL_S = 3600  # conversation history TTL (1 hour)

    def __init__(
        self,
        session_id: str,
        scenario: str = "hospital",
        redis=None,
    ) -> None:
        self._session_id = session_id
        self._scenario = scenario
        self._redis = redis
        self._history: List[dict] = []   # in-memory mirror
        self._client: Optional[anthropic.AsyncAnthropic] = None
        self._error_context: List[str] = []   # recent sign errors for personalisation

    async def start(self) -> None:
        if not settings.ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY not set — LLM will use mock responses")
            return
        self._client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        # Restore history from Redis if session was interrupted
        if self._redis:
            raw = await self._redis.get(f"history:{self._session_id}")
            if raw:
                self._history = json.loads(raw)

    async def chat(
        self,
        user_sign_text: str,
        error_hints: Optional[List[str]] = None,
    ) -> LLMResponse:
        """
        Send a recognised sign (as Korean text) to the LLM and get a response.

        Parameters
        ----------
        user_sign_text : Korean word/sentence the user signed, e.g. "안녕하세요"
        error_hints    : List of short error strings from the feedback engine,
                         e.g. ["엄지 방향 틀림", "손목 각도 낮음"]
        """
        if error_hints:
            self._error_context.extend(error_hints)
            self._error_context = self._error_context[-5:]   # keep last 5

        user_content = user_sign_text
        if error_hints:
            user_content += f"\n[인식된 오류: {', '.join(error_hints)}]"

        self._history.append({"role": "user", "content": user_content})

        if self._client is None:
            return self._mock_response(user_sign_text)

        system_prompt = self._build_system_prompt()

        try:
            response = await self._client.messages.create(
                model=settings.LLM_MODEL,
                max_tokens=settings.LLM_MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        # Cache the system prompt — saves ~200 ms on subsequent turns
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=self._history[-self.MAX_HISTORY:],
            )

            reply_text = response.content[0].text.strip()
            tokens = (
                response.usage.input_tokens + response.usage.output_tokens
                if hasattr(response, "usage")
                else 0
            )
        except anthropic.APIError as e:
            logger.error("Anthropic API error: %s", e)
            reply_text = "죄송합니다. 잠시 후 다시 시도해주세요."
            tokens = 0

        self._history.append({"role": "assistant", "content": reply_text})

        # Persist to Redis
        if self._redis:
            await self._redis.setex(
                f"history:{self._session_id}",
                self.CACHE_TTL_S,
                json.dumps(self._history[-self.MAX_HISTORY:]),
            )

        from app.services.avatar.controller import AvatarController
        avatar_commands = AvatarController.text_to_commands(reply_text)

        return LLMResponse(
            text=reply_text,
            avatar_commands=avatar_commands,
            tokens_used=tokens,
        )

    async def stop(self) -> None:
        if self._client:
            await self._client.close()

    # ─────────────────────── helpers ────────────────────────────

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
        """Used when no API key is configured."""
        responses = {
            "안녕하세요": "안녕하세요! 오늘도 수어 연습 잘 하고 있네요. 다음 단어를 해볼까요?",
            "감사합니다": "천만에요! 감사 표현 수어가 정확해졌어요.",
            "아프다": "아픈 곳이 있나요? '아프다' 수어를 잘 표현했어요.",
            "약": "약이 필요하신가요? 간호사에게 말씀드릴게요.",
            "의사": "의사 선생님을 찾고 계신가요? 안내해 드릴게요.",
            "도와주세요": "도움이 필요하신가요? 도와드리겠습니다!",
        }
        text = responses.get(user_text, f"'{user_text}' 수어를 잘 표현했어요. 계속 연습해보세요!")
        from app.services.avatar.controller import AvatarController
        return LLMResponse(
            text=text,
            avatar_commands=AvatarController.text_to_commands(text),
            tokens_used=0,
        )
