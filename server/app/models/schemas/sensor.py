"""
Pydantic schemas for sensor data.

VisionData   : MediaPipe hand landmark output (21 points × 3D)
GloveData    : ESP32 BLE packet (5 flex sensors + IMU)
SensorFrame  : One timestamped, fused snapshot sent over WebSocket

When the real glove is connected, GloveData.is_mock will be False
and ble_quality will reflect actual RSSI / packet-loss metrics.
"""

from __future__ import annotations

import time
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class HandLandmark(BaseModel):
    x: float
    y: float
    z: float


class VisionData(BaseModel):
    # 21 MediaPipe hand landmarks in image-normalised coordinates
    landmarks: List[HandLandmark] = Field(min_length=21, max_length=21)
    # Same 21 points in metric (world) coordinates — used for feedback 3-D replay
    world_landmarks: List[HandLandmark] = Field(min_length=21, max_length=21)
    confidence: float = Field(ge=0.0, le=1.0)
    handedness: str = "Right"   # "Left" | "Right"
    # Frames-per-second the client is currently delivering
    fps: float = 30.0

    @field_validator("handedness")
    @classmethod
    def _valid_hand(cls, v: str) -> str:
        if v not in ("Left", "Right"):
            raise ValueError("handedness must be 'Left' or 'Right'")
        return v


class IMUData(BaseModel):
    # Linear acceleration in m/s²  [x, y, z]
    accel: List[float] = Field(min_length=3, max_length=3)
    # Angular velocity in rad/s    [x, y, z]
    gyro: List[float] = Field(min_length=3, max_length=3)
    # Orientation quaternion       [w, x, y, z]
    quaternion: List[float] = Field(min_length=4, max_length=4)


class GloveData(BaseModel):
    # Normalised flex-sensor readings  0.0 = fully open, 1.0 = fully closed
    # Order: [thumb, index, middle, ring, pinky]
    flex: List[float] = Field(min_length=5, max_length=5)
    imu: IMUData
    # Derived from BLE RSSI + packet-loss ratio.  1.0 = perfect, 0.0 = unusable.
    ble_quality: float = Field(ge=0.0, le=1.0, default=1.0)
    # Monotonically increasing counter on the ESP32 side — lets us detect drops
    sequence: int = 0
    # True while no real hardware is present (mock data injected by server)
    is_mock: bool = False


class SensorFrame(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    # Frame counter assigned by the client (camera fps reference)
    sequence: int
    session_id: str
    camera: Optional[VisionData] = None        # 오른손
    camera_left: Optional[VisionData] = None   # 왼손
    glove: Optional[GloveData] = None          # 오른손 글러브
    glove_left: Optional[GloveData] = None     # 왼손 글러브
