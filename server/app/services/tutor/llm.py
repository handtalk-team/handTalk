"""
LLM 호출 래퍼 — 프로바이더(ollama / groq / claude)를 감춘다.
LLM_PROVIDER 설정이 없거나 패키지/키가 없으면 available=False 이고, 튜터는 규칙 기반으로 동작한다.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class LLMClient:
    def __init__(self) -> None:
        self.provider = settings.LLM_PROVIDER
        self._client = None
        try:
            if self.provider in ("ollama", "groq"):
                from openai import AsyncOpenAI
                if self.provider == "ollama":
                    self._client = AsyncOpenAI(base_url=settings.OLLAMA_BASE_URL, api_key="ollama")
                elif settings.GROQ_API_KEY:
                    self._client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1",
                                               api_key=settings.GROQ_API_KEY)
            elif self.provider == "claude" and settings.ANTHROPIC_API_KEY:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        except ImportError as e:
            logger.warning("LLM 패키지 없음 (%s) — 규칙 기반으로 진행", e)
        logger.info("LLM provider=%s available=%s", self.provider, self.available)

    @property
    def available(self) -> bool:
        return self._client is not None

    async def complete(self, system: str, messages: List[dict], max_tokens: int = 200,
                       temperature: float = 0.7) -> Optional[str]:
        """messages: [{"role": "user"|"assistant", "content": str}]. 실패하면 None."""
        if not self._client:
            return None
        try:
            if self.provider == "claude":
                resp = await self._client.messages.create(
                    model=settings.ANTHROPIC_MODEL, max_tokens=max_tokens,
                    system=system, messages=messages)
                return resp.content[0].text.strip()
            model = settings.GROQ_MODEL if self.provider == "groq" else settings.OLLAMA_MODEL
            resp = await self._client.chat.completions.create(
                model=model, max_tokens=max_tokens, temperature=temperature,
                messages=[{"role": "system", "content": system}, *messages])
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error("LLM 호출 실패: %s", e)
            return None

    async def close(self) -> None:
        if self._client and hasattr(self._client, "close"):
            try:
                await self._client.close()
            except Exception:
                pass


def is_korean_only(text: str) -> bool:
    """일본어/한자가 섞이거나 영어 비율이 높으면 False (소형 모델 오류 방지)"""
    if not text:
        return False
    for ch in text:
        c = ord(ch)
        if 0x3040 <= c <= 0x30FF or 0x3400 <= c <= 0x4DBF or 0x4E00 <= c <= 0x9FFF:
            return False
    ascii_alpha = sum(1 for c in text if ord(c) < 128 and c.isalpha())
    total_alpha = sum(1 for c in text if c.isalpha())
    return not (total_alpha and ascii_alpha / total_alpha > 0.3)
