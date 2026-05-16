"""
Vision Capture Module
=====================

WebcamVisionCapture  — grabs frames from the laptop webcam (cv2),
                       runs MediaPipe Hands, and produces VisionData.
                       Used NOW while the web client isn't yet integrated.

ClientVisionCapture  — receives MediaPipe landmarks that were already
                       computed in the browser (lighter WebSocket payload).
                       This is the production path: the client sends a
                       FrameMessage with VisionData already filled in, and
                       the server skips re-running MediaPipe.

Switch between them in the engine by setting USE_LOCAL_CAMERA in .env.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import List, Optional

import cv2
import mediapipe as mp

from app.models.schemas.sensor import HandLandmark, VisionData

mp_hands = mp.solutions.hands


class VisionCaptureInterface(ABC):
    @abstractmethod
    async def read_frame(self) -> Optional[VisionData]:
        """Return the latest processed frame, or None if no hand detected."""
        ...

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...


# ─────────────────────── local webcam path ──────────────────────


class WebcamVisionCapture(VisionCaptureInterface):
    """
    Captures from the laptop webcam in a background thread so it never
    blocks the async event loop.

    MediaPipe runs in the same thread as OpenCV (CPU-bound), but we offload
    the entire thing to asyncio's thread-pool executor.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        self._index = camera_index
        self._width = width
        self._height = height
        self._fps = fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._hands: Optional[mp_hands.Hands] = None
        self._latest: Optional[VisionData] = None
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._init_capture)
        self._task = asyncio.create_task(self._capture_loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._release)

    async def read_frame(self) -> Optional[VisionData]:
        async with self._lock:
            return self._latest

    # ── internals ────────────────────────────────────────────────

    def _init_capture(self) -> None:
        self._cap = cv2.VideoCapture(self._index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _release(self) -> None:
        if self._cap:
            self._cap.release()
        if self._hands:
            self._hands.close()

    async def _capture_loop(self) -> None:
        loop = asyncio.get_event_loop()
        period = 1.0 / self._fps
        while True:
            t0 = time.monotonic()
            vision_data = await loop.run_in_executor(None, self._process_one_frame)
            async with self._lock:
                self._latest = vision_data
            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0.0, period - elapsed))

    def _process_one_frame(self) -> Optional[VisionData]:
        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if not ret:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None

        hand_landmarks = result.multi_hand_landmarks[0]
        world_landmarks = result.multi_hand_world_landmarks[0]
        handedness = (
            result.multi_handedness[0].classification[0].label
            if result.multi_handedness
            else "Right"
        )
        score = (
            result.multi_handedness[0].classification[0].score
            if result.multi_handedness
            else 0.9
        )

        def _to_list(lm_list) -> List[HandLandmark]:
            return [
                HandLandmark(x=lm.x, y=lm.y, z=lm.z)
                for lm in lm_list.landmark
            ]

        return VisionData(
            landmarks=_to_list(hand_landmarks),
            world_landmarks=_to_list(world_landmarks),
            confidence=float(score),
            handedness=handedness,
            fps=float(self._fps),
        )


# ─────────────── client-provided landmarks (production) ─────────


class ClientVisionCapture(VisionCaptureInterface):
    """
    The web client runs MediaPipe in the browser and sends landmarks
    inside FrameMessage.data.camera.  The server just stores and reads
    the pre-computed VisionData.

    This class is a thin in-memory buffer so the engine can use the
    same interface regardless of capture source.
    """

    def __init__(self) -> None:
        self._latest: Optional[VisionData] = None

    async def start(self) -> None:
        pass  # nothing to initialise

    async def stop(self) -> None:
        pass

    async def read_frame(self) -> Optional[VisionData]:
        return self._latest

    def update(self, vision_data: Optional[VisionData]) -> None:
        """Called by the WebSocket handler whenever a new frame arrives."""
        self._latest = vision_data
