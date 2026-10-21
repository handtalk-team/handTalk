"""
WebSocket Connection Manager
============================
Tracks all active WebSocket connections (one per browser tab / device).
Provides broadcast helpers and per-session send wrappers.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Dict, Optional, Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self) -> None:
        # session_id → set of websockets (supports multiple tabs)
        self._sessions: Dict[str, Set[WebSocket]] = defaultdict(set)
        # Unreal MetaHuman 클라이언트 (/ws/avatar) — 튜터 응답을 받기만 한다
        self._avatars: Set[WebSocket] = set()
        self._motion_q: Optional[asyncio.Queue] = None
        self._motion_task: Optional[asyncio.Task] = None
        # 마지막 학습 목표 — 아바타 화면이 수업 시작보다 늦게 붙어도 시범을 보여 주려고 기억한다
        self._last_lesson: Optional[dict] = None

    async def accept(self, ws: WebSocket) -> None:
        await ws.accept()

    async def connect(self, ws: WebSocket, session_id: str) -> None:
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

    def add_avatar(self, ws: WebSocket) -> None:
        self._avatars.add(ws)
        logger.info("Avatar client connected  total=%d", len(self._avatars))

    def remove_avatar(self, ws: WebSocket) -> None:
        self._avatars.discard(ws)
        logger.info("Avatar client disconnected  total=%d", len(self._avatars))

    async def broadcast_avatar(self, data: dict) -> None:
        await self._send_avatars(data)
        if data.get("type") == "lesson":
            self._last_lesson = data
        # 튜터 답/학습 목표는 실시간 동작으로도 만들어 보낸다 (순서를 지키려고 한 줄로 처리)
        if data.get("type") in ("llm_response", "lesson") and self._avatars:
            self._ensure_motion_worker()
            self._motion_q.put_nowait(data)

    async def send_rest(self, ws: WebSocket) -> None:
        """새로 붙은 아바타를 지금 수업의 반복 시범으로 (수업이 없으면 쉬는 자세로)"""
        from app.services.avatar.motion_synth import get_synth, motion_for
        synth = await asyncio.to_thread(get_synth)
        msg = None
        if self._last_lesson is not None:
            msg = await asyncio.to_thread(motion_for, self._last_lesson)
        if msg is None and synth:
            msg = synth.rest()
        if msg:
            await ws.send_json(msg)

    def _ensure_motion_worker(self) -> None:
        if self._motion_task is None or self._motion_task.done():
            self._motion_q = asyncio.Queue()
            self._motion_task = asyncio.create_task(self._motion_worker())

    async def _motion_worker(self) -> None:
        from app.services.avatar.motion_synth import motion_for
        while True:
            data = await self._motion_q.get()
            try:
                msg = await asyncio.to_thread(motion_for, data)
                if msg:
                    await self._send_avatars(msg)
            except Exception:
                logger.exception("실시간 동작 생성 실패")

    async def _send_avatars(self, data: dict) -> None:
        dead: list[WebSocket] = []
        for ws in self._avatars:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._avatars.discard(ws)

    @property
    def total(self) -> int:
        return sum(len(v) for v in self._sessions.values())


manager = ConnectionManager()
