"""
WebSocket Session Handler
==========================

One coroutine per connected client.  Lifecycle:

  connect
    → start_session  (client sends StartSession JSON)
    → frame loop     (client sends FrameMessage JSON at ~30 Hz)
        → sensor fusion   → recognition engine
        → if recognised: LLM pipeline → avatar commands
        → feedback engine (inline)
    → end_session    (client sends EndSession or disconnects)
        → session summary → persist to DB
  disconnect

Latency monitoring
------------------
Each frame logs processing time.  If the rolling average exceeds
LATENCY_WARN_MS the server emits a "system/warning" to the client.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

from app.api.ws.manager import manager
from app.core.config import get_settings
from app.core.redis_client import get_redis
from app.models.schemas.sensor import SensorFrame
from app.models.schemas.ws_messages import (
    EndSession,
    FrameMessage,
    RecognitionResult,
    SessionSummary,
    StartSession,
    SystemMessage,
)
from app.services.feedback.engine import FeedbackEngine
from app.services.llm.pipeline import LLMPipeline
from app.services.recognition.engine import HybridRecognitionEngine
from app.services.recognition.glove import MockGloveSensor
from app.services.recognition.vision import ClientVisionCapture, WebcamVisionCapture

logger = logging.getLogger(__name__)
settings = get_settings()

LATENCY_WARN_MS = 400   # warn if a single frame takes > 400 ms end-to-end


class SessionHandler:
    """Manages the full lifecycle of one WebSocket connection."""

    def __init__(self, ws: WebSocket) -> None:
        self._ws = ws
        self._session_id: Optional[str] = None
        self._scenario: str = "hospital"
        self._user_id: Optional[str] = None

        self._engine = HybridRecognitionEngine()
        self._feedback = FeedbackEngine()
        self._llm: Optional[LLMPipeline] = None

        # Glove source: mock (now) or real BLE (future)
        self._glove = MockGloveSensor(hz=settings.MOCK_GLOVE_HZ)

        # Vision source: local webcam or client-provided landmarks
        self._vision: ClientVisionCapture | WebcamVisionCapture

        # Latency tracking (rolling 30-frame window)
        self._latencies: list[float] = []

        self._running = False

    # ─────────────────────── lifecycle ──────────────────────────

    async def run(self) -> None:
        """Entry point called from the FastAPI WebSocket route."""
        # Assign a temporary session ID immediately so we can accept the socket
        temp_id = str(uuid.uuid4())
        await manager.connect(self._ws, temp_id)
        self._session_id = temp_id

        try:
            await manager.accept(self._ws)
            await self._send(SystemMessage(
                level="info",
                message="연결되었습니다. 세션을 시작하려면 start_session 메시지를 보내주세요.",
            ))
            await self._main_loop()
        except WebSocketDisconnect:
            logger.info("Client disconnected session=%s", self._session_id)
        except Exception as e:
            logger.exception("Unhandled error in session %s: %s", self._session_id, e)
            await self._send(SystemMessage(level="error", message=str(e)))
        finally:
            await self._cleanup()
            manager.disconnect(self._ws, self._session_id)

    # ─────────────────────── main loop ──────────────────────────

    async def _main_loop(self) -> None:
        while True:
            raw = await self._ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await self._send(SystemMessage(level="error", message="Invalid JSON"))
                continue

            msg_type = msg.get("type")

            if msg_type == "start_session":
                await self._handle_start(StartSession(**msg))
            elif msg_type == "frame":
                if self._running:
                    await self._handle_frame(msg)
            elif msg_type == "end_session":
                await self._handle_end()
                break
            else:
                await self._send(SystemMessage(
                    level="warning",
                    message=f"Unknown message type: {msg_type}",
                ))

    # ─────────────────────── handlers ───────────────────────────

    async def _handle_start(self, msg: StartSession) -> None:
        self._scenario = msg.scenario
        self._user_id = msg.user_id

        # Reassign session ID (now we have user context)
        manager.disconnect(self._ws, self._session_id)
        self._session_id = str(uuid.uuid4())
        await manager.connect(self._ws, self._session_id)

        # Start sensor sources
        await self._glove.start()
        self._vision = ClientVisionCapture()    # browser sends landmarks

        # Start feedback reference loading (non-blocking)
        asyncio.create_task(self._load_feedback_refs())

        # Start LLM pipeline
        redis = await get_redis()
        self._llm = LLMPipeline(
            session_id=self._session_id,
            scenario=self._scenario,
            redis=redis,
        )
        await self._llm.start()

        self._running = True
        await self._send(SystemMessage(
            level="info",
            message=f"세션 시작! 시나리오: {self._scenario}  ID: {self._session_id}",
        ))

    async def _handle_frame(self, raw_msg: dict) -> None:
        t0 = time.perf_counter()

        # Inject server-generated glove data if client didn't provide any
        frame_data = raw_msg.get("data", {})
        if not frame_data.get("glove"):
            mock_glove = await self._glove.read()
            frame_data["glove"] = mock_glove.model_dump()

        # Ensure session_id is set
        frame_data.setdefault("session_id", self._session_id)
        frame_data.setdefault("sequence", 0)

        # Update client-provided vision data in the capture buffer
        if isinstance(self._vision, ClientVisionCapture):
            from app.models.schemas.sensor import VisionData
            cam_data = frame_data.get("camera")
            if cam_data:
                self._vision.update(VisionData(**cam_data))

        try:
            sensor_frame = SensorFrame(**frame_data)
        except Exception as e:
            logger.warning("Bad SensorFrame: %s", e)
            return

        # ── Recognition ──────────────────────────────────────────
        result: Optional[RecognitionResult] = await self._engine.process_frame(
            sensor_frame
        )

        if result:
            await self._send(result)

            if not result.is_partial:
                # ── LLM response ──────────────────────────────────────
                llm_resp = await self._llm.chat(result.text)
                await self._send(llm_resp)

                # ── Inline feedback ───────────────────────────────────
                window = self._engine._fusion.get_window_array()
                if window is not None:
                    fb = self._feedback.analyse(
                        sign_label=result.text,
                        feature_window=window,
                        is_correct=result.confidence >= settings.RECOGNITION_THRESHOLD,
                        confidence=result.confidence,
                    )
                    if fb.errors:
                        await self._send(fb)

        # ── Latency monitoring ────────────────────────────────────
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._latencies.append(elapsed_ms)
        if len(self._latencies) > 30:
            self._latencies.pop(0)
        avg_ms = sum(self._latencies) / len(self._latencies)
        if avg_ms > LATENCY_WARN_MS:
            await self._send(SystemMessage(
                level="warning",
                message=f"처리 지연 경고: 평균 {avg_ms:.0f}ms (목표 < {LATENCY_WARN_MS}ms)",
            ))

    async def _handle_end(self) -> None:
        self._running = False
        summary: SessionSummary = self._feedback.build_session_summary(
            self._session_id
        )
        await self._send(summary)
        logger.info(
            "Session ended: %s  signs=%d  accuracy=%.1f%%",
            self._session_id,
            summary.total_signs,
            summary.accuracy * 100,
        )

    # ─────────────────────── helpers ────────────────────────────

    async def _send(self, msg) -> None:
        await manager.send(self._ws, msg.model_dump())

    async def _load_feedback_refs(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._feedback.load_references)

    async def _cleanup(self) -> None:
        self._running = False
        await self._glove.stop()
        if self._llm:
            await self._llm.stop()
        self._engine.reset_session()


# ─────────────────── FastAPI route entry-point ───────────────────


async def websocket_endpoint(ws: WebSocket) -> None:
    handler = SessionHandler(ws)
    await handler.run()
