"""
WebSocket Connection Manager
============================
Tracks all active WebSocket connections (one per browser tab / device).
Provides broadcast helpers and per-session send wrappers.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self) -> None:
        # session_id → set of websockets (supports multiple tabs)
        self._sessions: Dict[str, Set[WebSocket]] = defaultdict(set)

    async def connect(self, ws: WebSocket, session_id: str) -> None:
        await ws.accept()
        self._sessions[session_id].add(ws)
        logger.info("WS connected   session=%s  total=%d", session_id, self.total)

    def disconnect(self, ws: WebSocket, session_id: str) -> None:
        self._sessions[session_id].discard(ws)
        if not self._sessions[session_id]:
            del self._sessions[session_id]
        logger.info("WS disconnected session=%s  total=%d", session_id, self.total)

    async def send(self, ws: WebSocket, data: dict) -> None:
        try:
            await ws.send_json(data)
        except Exception as e:
            logger.warning("WS send failed: %s", e)

    async def broadcast_to_session(self, session_id: str, data: dict) -> None:
        dead: list[WebSocket] = []
        for ws in self._sessions.get(session_id, set()):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._sessions[session_id].discard(ws)

    @property
    def total(self) -> int:
        return sum(len(v) for v in self._sessions.values())


manager = ConnectionManager()
