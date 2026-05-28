"""
Vision Capture Module (MediaPipe Tasks API — mediapipe >= 0.10)
=======================================================

WebcamVisionCapture  — 노트북 웹캠(cv2) + MediaPipe HandLandmarker Tasks API
                       서버에서 직접 캡처할 때 사용 (현재 기본값)

ClientVisionCapture  — 브라우저에서 이미 계산된 랜드마크를 수신
                       (웹 클라이언트가 MediaPipe JS를 사용할 때 프로덕션 경로)

모델 파일: ml/models/hand_landmarker.task
  없으면 자동 다운로드를 시도합니다.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import urllib.request
from abc import ABC, abstractmethod
from typing import List, Optional

import cv2

from app.models.schemas.sensor import HandLandmark, VisionData

logger = logging.getLogger(__name__)

MODEL_PATH = "ml/models/hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def _ensure_model() -> str:
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        logger.info("HandLandmarker 모델 다운로드 중: %s", MODEL_URL)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        logger.info("다운로드 완료: %s", MODEL_PATH)
    return MODEL_PATH


class VisionCaptureInterface(ABC):
    @abstractmethod
    async def read_frame(self) -> Optional[VisionData]:
        ...

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...


# ─────────────────────── 로컬 웹캠 캡처 ─────────────────────────


class WebcamVisionCapture(VisionCaptureInterface):
    """
    노트북 웹캠에서 직접 캡처하고 MediaPipe Tasks API로 랜드마크를 추출합니다.
    백그라운드 스레드에서 실행되어 async 이벤트 루프를 블로킹하지 않습니다.
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
        self._landmarker = None
        self._latest: Optional[VisionData] = None
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._init)
        self._task = asyncio.create_task(self._loop())

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

    # ── 내부 ─────────────────────────────────────────────────────

    def _init(self) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = _ensure_model()
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

        self._cap = cv2.VideoCapture(self._index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        logger.info("웹캠 초기화 완료 (index=%d)", self._index)

    def _release(self) -> None:
        if self._cap:
            self._cap.release()
        if self._landmarker:
            self._landmarker.close()

    async def _loop(self) -> None:
        loop = asyncio.get_event_loop()
        period = 1.0 / self._fps
        while True:
            t0 = time.monotonic()
            result = await loop.run_in_executor(None, self._process_frame)
            async with self._lock:
                self._latest = result
            await asyncio.sleep(max(0.0, period - (time.monotonic() - t0)))

    def _process_frame(self) -> Optional[VisionData]:
        import mediapipe as mp

        if not self._cap or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection = self._landmarker.detect(mp_image)

        if not detection.hand_landmarks:
            return None

        lms  = detection.hand_landmarks[0]       # 21개 이미지 좌표
        wlms = detection.hand_world_landmarks[0] if detection.hand_world_landmarks else lms
        hand = detection.handedness[0][0].category_name if detection.handedness else "Right"
        conf = detection.handedness[0][0].score   if detection.handedness else 0.9

        def _to_list(landmark_list) -> List[HandLandmark]:
            return [HandLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in landmark_list]

        return VisionData(
            landmarks=_to_list(lms),
            world_landmarks=_to_list(wlms),
            confidence=float(conf),
            handedness=hand if hand in ("Left", "Right") else "Right",
            fps=float(self._fps),
        )


# ──────── 클라이언트가 제공하는 랜드마크 (프로덕션) ─────────────


class ClientVisionCapture(VisionCaptureInterface):
    """
    브라우저가 MediaPipe JS로 계산한 랜드마크를 FrameMessage로 전송합니다.
    서버는 이 클래스를 통해 수신된 VisionData를 그대로 저장·반환합니다.
    """

    def __init__(self) -> None:
        self._latest: Optional[VisionData] = None        # 오른손
        self._latest_left: Optional[VisionData] = None   # 왼손

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def read_frame(self) -> Optional[VisionData]:
        return self._latest

    def update(self, vision_data: Optional[VisionData]) -> None:
        self._latest = vision_data

    def update_left(self, vision_data: Optional[VisionData]) -> None:
        self._latest_left = vision_data
