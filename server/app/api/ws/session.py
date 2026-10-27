"""
WebSocket Session Handler
==========================

One coroutine per connected client.  Lifecycle:

  connect
    → start_session  (client sends StartSession JSON)
    → frame loop     (client sends FrameMessage JSON at ~30 Hz)
        → sensor fusion   → recognition engine
        → if recognised: mode tutor (word / sentence / free_talk) → avatar commands
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
import os
import time
import uuid
from typing import Optional

import numpy as np

from fastapi import WebSocket, WebSocketDisconnect

from app.api.ws.manager import manager
from app.core.config import get_settings
from app.core.database import AsyncSessionLocal
from app.core.redis_client import get_redis
from app.models.domain.session import DataSample, LearningSession, SignAttempt
from app.models.schemas.sensor import SensorFrame
from app.models.schemas.ws_messages import (
    CollectAck,
    EndSession,
    FrameMessage,
    GloveStatus,
    ModalityWeights,
    RecognitionResult,
    SessionSummary,
    StartSession,
    SystemMessage,
)
from app.services.feedback.engine import FeedbackEngine
from app.services.tutor.llm import LLMClient
from app.services.tutor.tutors import BaseTutor, make_tutor
from app.services.recognition.engine import HybridRecognitionEngine
from app.services.recognition.sentence import SentenceRecognizer
from app.services.recognition.glove import (
    AutoGloveHub,
    AutoGloveSensor,
    GloveSensorInterface,
    SerialGloveSensor,
)
from app.services.recognition.vision import ClientVisionCapture, WebcamVisionCapture

logger = logging.getLogger(__name__)
settings = get_settings()

LATENCY_WARN_MS = 400   # warn if a single frame takes > 400 ms end-to-end

# ── 시리얼 글러브 싱글턴 ─────────────────────────────────────────────
# 시리얼 포트는 프로세스 전체에서 하나만 열 수 있으므로 모듈 레벨에서 공유
# 포트를 지정하지 않으면 USB 글러브를 자동 감지 (꽂으면 붙고, 없으면 카메라만)
_glove_hub = AutoGloveHub(settings.GLOVE_BAUD_RATE)
_glove_right: GloveSensorInterface = (
    SerialGloveSensor(settings.GLOVE_PORT_RIGHT, settings.GLOVE_BAUD_RATE, "R")
    if settings.GLOVE_PORT_RIGHT else AutoGloveSensor(_glove_hub, "R")
)
_glove_left: GloveSensorInterface = (
    SerialGloveSensor(settings.GLOVE_PORT_LEFT, settings.GLOVE_BAUD_RATE, "L")
    if settings.GLOVE_PORT_LEFT else AutoGloveSensor(_glove_hub, "L")
)
_glove_started = False


async def _ensure_gloves_started() -> None:
    global _glove_started
    logger.info("_ensure_gloves_started called, already_started=%s, right_type=%s, port=%s",
                _glove_started, type(_glove_right).__name__, getattr(_glove_right, '_port', 'N/A'))
    if not _glove_started:
        await _glove_right.start()
        await _glove_left.start()
        _glove_started = True
        logger.info("Glove sensors started: right_running=%s left_running=%s",
                    getattr(_glove_right, '_running', '?'), getattr(_glove_left, '_running', '?'))


class SessionHandler:
    """Manages the full lifecycle of one WebSocket connection."""

    def __init__(self, ws: WebSocket) -> None:
        self._ws = ws
        self._session_id: Optional[str] = None
        self._scenario: str = "hospital"
        self._user_id: Optional[str] = None

        self._engine = HybridRecognitionEngine()
        self._sentence = SentenceRecognizer()
        self._feedback = FeedbackEngine()
        self._mode: str = "word"
        self._llm: Optional[LLMClient] = None
        self._tutor: Optional[BaseTutor] = None

        # 글러브 싱글턴 참조 (포트는 모듈 로드 시 한 번만 열림)
        self._glove      = _glove_right
        self._glove_left = _glove_left

        # Vision source: local webcam or client-provided landmarks
        self._vision: ClientVisionCapture | WebcamVisionCapture

        # Latency tracking (rolling 30-frame window)
        self._latencies: list[float] = []

        self._running = False
        self._recording = False
        self._glove_frame_counter = 0
        self._ws_active = True   # WebSocket 연결 동안 True, 종료 시 False

    # ─────────────────────── lifecycle ──────────────────────────

    async def run(self) -> None:
        """Entry point called from the FastAPI WebSocket route."""
        await self._ws.accept()

        self._session_id = str(uuid.uuid4())
        await manager.connect(self._ws, self._session_id)

        # 글러브 센서 시작 (이미 시작됐으면 no-op)
        await _ensure_gloves_started()
        asyncio.create_task(self._glove_broadcast_loop())

        try:
            await self._send(SystemMessage(
                level="info",
                message="연결되었습니다. 세션을 시작하려면 start_session 메시지를 보내주세요.",
            ))
            await self._main_loop()
        except WebSocketDisconnect:
            logger.info("Client disconnected session=%s", self._session_id)
        except Exception as e:
            logger.exception("Unhandled error in session %s: %s", self._session_id, e)
            try:
                await self._send(SystemMessage(level="error", message=str(e)))
            except Exception:
                pass
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
            elif msg_type == "recording_mode":
                active = msg.get("active", False)
                self._recording = active
                if active:
                    # 녹화 시작 시 버퍼 초기화 — 이전 추론 잔여 데이터 제거
                    self._engine._fusion.reset()
            elif msg_type == "capture_sample":
                self._recording = False
                label = msg.get("label", "").strip()
                if label:
                    await self._capture_sample(label)
                else:
                    await self._send(SystemMessage(level="warning", message="레이블을 입력해주세요."))
            elif msg_type == "tutor_command":
                if self._running and self._tutor:
                    for out in await self._tutor.on_command(msg.get("action", "")):
                        await self._send(out)
            elif msg_type == "end_session":
                if self._running:
                    await self._handle_end()
                    await self._between_sessions_cleanup()
                # break 하지 않음 — 연결 유지, start_session 재사용 가능
            else:
                await self._send(SystemMessage(
                    level="warning",
                    message=f"Unknown message type: {msg_type}",
                ))

    # ─────────────────────── handlers ───────────────────────────

    async def _handle_start(self, msg: StartSession) -> None:
        # 재시작 시 새 세션 ID 발급
        self._session_id = str(uuid.uuid4())
        self._scenario = msg.scenario
        self._user_id = msg.user_id

        # Start vision source (glove already started on connect)
        self._vision = ClientVisionCapture()    # browser sends landmarks

        # Start feedback reference loading (non-blocking)
        asyncio.create_task(self._load_feedback_refs())

        # 모드별 튜터 (단어 연습 / 상황별 문장 / 프리토킹)
        self._mode = msg.mode
        self._llm = LLMClient()
        self._tutor = make_tutor(msg.mode, msg.scenario, self._llm, msg.words)
        self._tutor.emit = self._send_all

        self._running = True

        # DB에 세션 기록 생성
        asyncio.create_task(self._db_create_session())

        await self._send(SystemMessage(
            level="info",
            message=f"세션 시작! 모드: {self._mode} 과정: {self._scenario}  ID: {self._session_id}",
        ))

        for out in await self._tutor.opening():
            await self._send(out)

    async def _handle_frame(self, raw_msg: dict) -> None:
        t0 = time.perf_counter()

        # Inject server-generated glove data if client didn't provide any
        frame_data = raw_msg.get("data", {})

        # Ensure session_id is set
        frame_data.setdefault("session_id", self._session_id)
        frame_data.setdefault("sequence", 0)

        # Update client-provided vision data in the capture buffer
        if isinstance(self._vision, ClientVisionCapture):
            from app.models.schemas.sensor import VisionData
            cam_data = frame_data.get("camera")
            cam_left_data = frame_data.get("camera_left")
            if cam_data:
                self._vision.update(VisionData(**cam_data))
            if cam_left_data:
                self._vision.update_left(VisionData(**cam_left_data))

        # Inject glove readings from serial sensors if client didn't provide them
        glove_data = None
        glove_left_data = None
        if "glove" not in frame_data or frame_data["glove"] is None:
            glove_data = await self._glove.read()
            if glove_data:
                frame_data["glove"] = glove_data.model_dump()
        if "glove_left" not in frame_data or frame_data["glove_left"] is None:
            glove_left_data = await self._glove_left.read()
            if glove_left_data:
                frame_data["glove_left"] = glove_left_data.model_dump()

        try:
            sensor_frame = SensorFrame(**frame_data)
        except Exception as e:
            logger.warning("Bad SensorFrame: %s", e)
            return

        # ── 녹화 중: 추론 없이 버퍼만 채우고 종료 ────────────────
        if self._recording:
            self._engine._fusion.push_frame(sensor_frame)
            return

        # ── Recognition ──────────────────────────────────────────
        result: Optional[RecognitionResult] = await self._engine.process_frame(
            sensor_frame
        )

        # ── 문장 인식 (프리토킹·상황 문장): 손을 내리거나 멈추면 한 문장을 통째로 읽는다 ──
        sentence_only = False
        if self._sentence.available and self._tutor and self._mode in ("free_talk", "sentence"):
            sentence_only = self._tutor.wants_sentence_only(self._sentence.vocab)
            ff = getattr(self._engine, "last_fused", None)
            if ff is not None:
                gloss = self._sentence.push(ff.fused_features, bool(np.abs(ff.vision_features).sum() > 0),
                                            timestamp=ff.timestamp, allowed=self._tutor.allowed_glosses())
                if gloss:
                    await self._send(RecognitionResult(text=" ".join(gloss), confidence=1.0,
                                                       modality_weights=ModalityWeights(vision=1.0, glove=0.0)))
                    for out in await self._tutor.on_sentence(gloss):
                        await self._send(out)

        # 연습 중인 목표 단어가 상위 후보 안에 있으면 그 단어로 본다 (웹캠 입력은 학습 데이터와 달라 1등이 자주 빗나감)
        if result and self._tutor and not result.is_partial:
            exp = self._tutor.expected_gloss()
            if exp and exp != result.text and exp in result.candidates[:settings.TARGET_TOPK]:
                logger.info("target '%s' in top-%d (1st '%s') -> accepted", exp, settings.TARGET_TOPK, result.text)
                result.text = exp

        if result and not self._recording:
            await self._send(result)

            if not result.is_partial and not sentence_only:
                # ── DB 저장 ───────────────────────────────────────────
                asyncio.create_task(self._db_save_attempt(result))

                # ── 튜터 응답 (모드별) ────────────────────────────────
                for out in await self._tutor.on_sign(result):
                    await self._send(out)

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

        # ── 프리토킹: 잠깐 멈추면 말이 끝난 것으로 보고 튜터가 답한다 ──
        if self._tutor:
            for out in await self._tutor.tick():
                await self._send(out)

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
        asyncio.create_task(self._db_close_session(summary))
        logger.info(
            "Session ended: %s  signs=%d  accuracy=%.1f%%",
            self._session_id,
            summary.total_signs,
            summary.accuracy * 100,
        )

    # ─────────────────────── helpers ────────────────────────────

    async def _send_all(self, msgs: list) -> None:
        if self._running:
            for m in msgs:
                await self._send(m)

    async def _send(self, msg) -> None:
        data = msg.model_dump()
        await manager.send(self._ws, data)
        # 튜터 응답과 인식 결과는 Unreal 아바타 클라이언트에도 보낸다
        if data.get("type") in AVATAR_MESSAGE_TYPES:
            await manager.broadcast_avatar(data)

    async def _db_create_session(self) -> None:
        from datetime import datetime
        try:
            async with AsyncSessionLocal() as db:
                db.add(LearningSession(
                    id=self._session_id,
                    user_id=self._user_id,
                    scenario=self._scenario,
                    started_at=datetime.utcnow(),
                ))
                await db.commit()
        except Exception as e:
            logger.warning("DB session create failed: %s", e)

    async def _db_save_attempt(self, result: RecognitionResult) -> None:
        from datetime import datetime
        try:
            async with AsyncSessionLocal() as db:
                db.add(SignAttempt(
                    id=str(uuid.uuid4()),
                    session_id=self._session_id,
                    sign_label=result.text,
                    is_correct=1 if result.confidence >= settings.RECOGNITION_THRESHOLD else 0,
                    confidence=result.confidence,
                    timestamp=datetime.utcnow(),
                ))
                await db.commit()
        except Exception as e:
            logger.warning("DB attempt save failed: %s", e)

    async def _db_close_session(self, summary: SessionSummary) -> None:
        from datetime import datetime
        try:
            async with AsyncSessionLocal() as db:
                row = await db.get(LearningSession, self._session_id)
                if row:
                    row.ended_at = datetime.utcnow()
                    row.total_signs = summary.total_signs
                    row.correct_signs = summary.correct_signs
                    row.accuracy = summary.accuracy
                    await db.commit()
        except Exception as e:
            logger.warning("DB session close failed: %s", e)

    async def _capture_sample(self, label: str) -> None:
        window = self._engine._fusion.get_window_array()
        if window is None or len(window) < 10:
            await self._send(SystemMessage(level="warning", message="윈도우 데이터 부족 — 동작 후 저장하세요."))
            return
        out_dir = os.path.join("ml", "data", label)
        os.makedirs(out_dir, exist_ok=True)
        existing = [f for f in os.listdir(out_dir) if f.startswith("sample_") and f.endswith(".npy")]
        idx = len(existing)
        path = os.path.join(out_dir, f"sample_{idx:03d}.npy")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, np.save, path, window.astype(np.float32))
        count = idx + 1
        logger.info("capture label=%s path=%s count=%d frames=%d", label, path, count, len(window))

        # DB에 메타데이터 저장
        try:
            from datetime import datetime
            async with AsyncSessionLocal() as db:
                db.add(DataSample(
                    id=str(uuid.uuid4()),
                    label=label,
                    file_path=path,
                    frames=len(window),
                    collected_at=datetime.utcnow(),
                ))
                await db.commit()
        except Exception as e:
            logger.warning("DB sample save failed: %s", e)

        await self._send(CollectAck(label=label, count=count))

    async def _glove_broadcast_loop(self) -> None:
        """300ms마다 글러브 상태를 클라이언트에 전송 (세션 여부 무관)."""
        count = 0
        while self._ws_active:
            try:
                right = await self._glove.read()
                left  = await self._glove_left.read()
                if count % 10 == 0:  # 3초마다 로그
                    logger.info("glove_broadcast: right=%s left=%s ws_active=%s",
                                right.flex[:2] if right else None,
                                left.flex[:2]  if left  else None,
                                self._ws_active)
                count += 1
                await self._send(GloveStatus(
                    right=right.model_dump() if right else None,
                    left=left.model_dump()  if left  else None,
                ))
            except Exception as e:
                logger.warning("glove_broadcast error: %s", e)
            await asyncio.sleep(0.3)

    async def _load_feedback_refs(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._feedback.load_references)

    async def _between_sessions_cleanup(self) -> None:
        """end_session 후 연결은 유지하면서 이전 세션 리소스만 정리."""
        if self._llm:
            await self._llm.close()
            self._llm = None
        self._tutor = None
        self._engine.reset_session()
        # 글러브 센서는 WebSocket 연결 동안 계속 유지

    async def _cleanup(self) -> None:
        """WebSocket 연결 종료 시 최종 정리."""
        self._running = False
        self._ws_active = False   # broadcast loop 종료
        # 글러브 싱글턴은 서버 프로세스가 살아있는 한 유지 — stop하지 않음
        if self._llm:
            await self._llm.close()
        self._engine.reset_session()


# ─────────────────── FastAPI route entry-point ───────────────────


async def websocket_endpoint(ws: WebSocket) -> None:
    handler = SessionHandler(ws)
    await handler.run()


AVATAR_MESSAGE_TYPES = ("llm_response", "recognition", "lesson")


async def avatar_endpoint(ws: WebSocket) -> None:
    """Unreal MetaHuman 클라이언트용. 서버 → 클라이언트 단방향 (llm_response, recognition)."""
    await manager.accept(ws)
    manager.add_avatar(ws)
    try:
        await manager.send_rest(ws)
        while True:
            await ws.receive_text()   # 연결 유지용, 내용은 무시
    except WebSocketDisconnect:
        pass
    finally:
        manager.remove_avatar(ws)
