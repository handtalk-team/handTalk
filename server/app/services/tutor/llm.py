"""
LLM 호출 래퍼 — 프로바이더를 감춘다.

  claude_code  이 컴퓨터에 로그인된 Claude Code 를 요청마다 비대화형(-p)으로 실행 (API 키 없이, 시연용)
  claude       Anthropic API (ANTHROPIC_API_KEY, 정식 서비스용)
  ollama/groq  OpenAI 호환 API

LLM_PROVIDER 설정이 없거나 사용할 수 없으면 available=False 이고, 튜터는 규칙 기반으로 동작한다.
"""

from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
from typing import List, Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class LLMClient:
    def __init__(self) -> None:
        self.provider = settings.LLM_PROVIDER
        self._client = None
        self._cc_path: Optional[str] = None
        try:
            if self.provider == "claude_code":
                self._cc_path = find_claude_code()
                if self._cc_path:
                    self._client = "claude_code"
            elif self.provider in ("ollama", "groq"):
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
            if self.provider == "claude_code":
                return await self._claude_code(system, messages)
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

    async def _claude_code(self, system: str, messages: List[dict]) -> Optional[str]:
        """대화 기록을 한 프롬프트로 묶어 Claude Code 를 한 번 실행하고 답만 받는다 (도구 사용 없음)."""
        convo = "\n".join(f"{'튜터' if m['role'] == 'assistant' else '사용자'}: {m['content']}" for m in messages)
        # 지시문은 단어 목록(수천 개) 때문에 길어서 명령줄 대신 stdin 으로 보낸다
        prompt = (f"<지시>\n{system}\n</지시>\n\n<대화>\n{convo}\n</대화>\n\n"
                  "위 대화에 이어 튜터의 다음 답을 지시대로 출력하세요.")
        args = [self._cc_path, "-p", "--output-format", "json", "--max-turns", "1",
                "--system-prompt", "당신은 한국 수어 튜터입니다. 입력의 <지시>를 그대로 따르고, 정해진 형식으로만 답합니다.", "--disallowedTools", "Bash,Edit,Write,Read,WebFetch,WebSearch,Glob,Grep,Task"]
        model = settings.CLAUDE_CODE_MODEL
        if model:
            args += ["--model", model]
        proc = await asyncio.create_subprocess_exec(
            *args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd=os.path.expanduser("~"))
        try:
            out, err = await asyncio.wait_for(proc.communicate(prompt.encode("utf-8")), timeout=settings.CLAUDE_CODE_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            logger.error("Claude Code 응답 시간 초과")
            return None
        if proc.returncode != 0:
            logger.error("Claude Code 실패 (%s): %s", proc.returncode, (out or err).decode("utf-8", "replace")[:300])
            return None
        try:
            return str(json.loads(out.decode("utf-8")).get("result", "")).strip() or None
        except ValueError:
            return out.decode("utf-8", "replace").strip() or None

    async def close(self) -> None:
        if self._client and not isinstance(self._client, str) and hasattr(self._client, "close"):
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


def find_claude_code() -> Optional[str]:
    """CLAUDE_CODE_PATH, PATH 의 claude, 또는 Claude 데스크톱 앱에 들어 있는 최신 claude.exe"""
    path = settings.CLAUDE_CODE_PATH or os.environ.get("CLAUDE_CODE_PATH")
    if path and os.path.exists(path):
        return path
    import shutil
    found = shutil.which("claude")
    if found:
        return found
    appdata = os.environ.get("APPDATA", "")
    cands = glob.glob(os.path.join(appdata, "Claude", "claude-code", "*", "*", "claude.exe"))
    # 스토어(MSIX)판 Claude 앱은 AppData 가 패키지 폴더로 가려져 있어 앱 밖에서는 여기서만 보인다
    local = os.environ.get("LOCALAPPDATA", "")
    cands += glob.glob(os.path.join(local, "Packages", "Claude_*", "LocalCache", "Roaming", "Claude",
                                    "claude-code", "*", "*", "claude.exe"))
    return max(cands, key=os.path.getmtime) if cands else None
