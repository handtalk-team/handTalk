"""
Thin wrapper around redis.asyncio.

Used for:
  - Session state cache        (key: "session:{id}")
  - Conversation history cache (key: "history:{session_id}")
  - Pub/sub for multi-instance WebSocket fan-out (future)
"""

import time
from typing import Optional

import redis.asyncio as aioredis

from .config import get_settings

settings = get_settings()

_redis: Optional[aioredis.Redis] = None
_unavailable_until: float = 0.0   # 실패 시 재시도 억제 (30초)


async def get_redis() -> Optional[aioredis.Redis]:
    global _redis, _unavailable_until
    if _redis is not None:
        return _redis
    if time.time() < _unavailable_until:
        return None  # 최근 실패 — 재시도 억제
    try:
        client = await aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=0.3,  # 0.3초 안에 안 되면 없는 것
        )
        await client.ping()
        _redis = client
    except Exception:
        _unavailable_until = time.time() + 30  # 30초 동안 재시도 안 함
        return None
    return _redis


async def close_redis() -> None:
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None
