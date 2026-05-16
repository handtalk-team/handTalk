"""
Thin wrapper around redis.asyncio.

Used for:
  - Session state cache        (key: "session:{id}")
  - Conversation history cache (key: "history:{session_id}")
  - Pub/sub for multi-instance WebSocket fan-out (future)
"""

from typing import Optional

import redis.asyncio as aioredis

from .config import get_settings

settings = get_settings()

_redis: Optional[aioredis.Redis] = None


async def get_redis() -> Optional[aioredis.Redis]:
    global _redis
    if _redis is None:
        try:
            client = await aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=1,
            )
            await client.ping()
            _redis = client
        except Exception:
            return None
    return _redis


async def close_redis() -> None:
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None
