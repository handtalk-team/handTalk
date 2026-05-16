"""
Glove Sensor Interface
======================

Two implementations share the same async interface:

  MockGloveSensor   — runs on the server, generates realistic fake data.
                      Used NOW while the real glove hardware is not yet ready.

  BLEGloveSensor    — stub ready for the real ESP32.
                      Swap MockGloveSensor → BLEGloveSensor in services/recognition/engine.py
                      when hardware arrives.  No other code needs to change.

Mock data strategy
------------------
The mock simulates a handful of distinct hand shapes that roughly correspond
to the sign vocabulary.  Each shape slowly transitions to the next to mimic
a user performing signs in sequence.  Gaussian noise is added to all channels
so the fusion module sees realistic confidence-vs-quality variation.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

from app.models.schemas.sensor import GloveData, IMUData

# ───────────────────── abstract interface ───────────────────────


class GloveSensorInterface(ABC):
    """Common interface for mock and real glove sensors."""

    @abstractmethod
    async def read(self) -> GloveData:
        """Return the latest sensor packet (non-blocking)."""
        ...

    @abstractmethod
    async def start(self) -> None:
        """Initialise hardware / start background task."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Clean up resources."""
        ...


# ───────────────────── hand-shape library ───────────────────────

@dataclass
class _HandShape:
    """Canonical flex + IMU values for a named hand posture."""
    name: str
    # Flex values 0.0 (open) – 1.0 (closed), [thumb, index, middle, ring, pinky]
    flex: List[float]
    # Wrist tilt in degrees [roll, pitch, yaw]
    orientation_deg: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


_HAND_SHAPES = [
    _HandShape("open",      flex=[0.05, 0.05, 0.05, 0.05, 0.05]),
    _HandShape("fist",      flex=[0.95, 0.95, 0.95, 0.95, 0.95]),
    _HandShape("point",     flex=[0.90, 0.05, 0.90, 0.90, 0.90]),
    _HandShape("peace",     flex=[0.90, 0.05, 0.05, 0.90, 0.90]),
    _HandShape("ok",        flex=[0.70, 0.70, 0.10, 0.10, 0.10],
               orientation_deg=[10.0, -20.0, 5.0]),
    _HandShape("thumb_up",  flex=[0.05, 0.90, 0.90, 0.90, 0.90],
               orientation_deg=[0.0,  30.0, 0.0]),
    _HandShape("pinch",     flex=[0.60, 0.60, 0.10, 0.10, 0.10]),
    _HandShape("wave_mid",  flex=[0.10, 0.10, 0.10, 0.10, 0.10],
               orientation_deg=[0.0,  15.0, 30.0]),
    _HandShape("cup",       flex=[0.40, 0.50, 0.50, 0.50, 0.50]),
    _HandShape("l_shape",   flex=[0.05, 0.05, 0.90, 0.90, 0.90],
               orientation_deg=[0.0, -10.0, 0.0]),
]


def _deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def _euler_to_quat(roll: float, pitch: float, yaw: float) -> List[float]:
    """ZYX Euler (rad) → quaternion [w, x, y, z]."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [w, x, y, z]


# ───────────────────── mock implementation ──────────────────────


class MockGloveSensor(GloveSensorInterface):
    """
    Generates synthetic BLE glove packets at a fixed rate.

    The mock cycles through _HAND_SHAPES, smoothly interpolating between
    them so the recognition model sees continuous motion rather than
    abrupt jumps.  Gaussian noise is added to simulate real sensor noise.

    Replace this class with BLEGloveSensor once the real ESP32 is ready.
    """

    # Seconds to hold each shape before transitioning
    HOLD_S: float = 1.5
    # Seconds the transition lasts
    TRANSITION_S: float = 0.5
    # Noise std-dev for flex sensors (normalised 0-1 range)
    FLEX_NOISE: float = 0.015
    # Noise std-dev for accelerometer (m/s²)
    ACCEL_NOISE: float = 0.08
    # Noise std-dev for gyroscope (rad/s)
    GYRO_NOISE: float = 0.005

    def __init__(self, hz: int = 50) -> None:
        self._hz = hz
        self._period = 1.0 / hz
        self._sequence = 0
        self._shape_idx = 0
        self._phase_start = time.time()
        self._in_transition = False
        self._task: asyncio.Task | None = None
        self._latest: GloveData = self._make_packet(
            _HAND_SHAPES[0], _HAND_SHAPES[0], 1.0
        )

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def read(self) -> GloveData:
        return self._latest

    # ── background loop ──────────────────────────────────────────

    async def _loop(self) -> None:
        while True:
            t0 = time.monotonic()
            self._tick()
            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0.0, self._period - elapsed))

    def _tick(self) -> None:
        now = time.time()
        phase_elapsed = now - self._phase_start

        current_shape = _HAND_SHAPES[self._shape_idx]
        next_shape = _HAND_SHAPES[
            (self._shape_idx + 1) % len(_HAND_SHAPES)
        ]

        if not self._in_transition:
            if phase_elapsed >= self.HOLD_S:
                self._in_transition = True
                self._phase_start = now
                phase_elapsed = 0.0

        if self._in_transition:
            t = min(phase_elapsed / self.TRANSITION_S, 1.0)
            if t >= 1.0:
                self._shape_idx = (self._shape_idx + 1) % len(_HAND_SHAPES)
                self._in_transition = False
                self._phase_start = now
                t = 1.0
            self._latest = self._make_packet(current_shape, next_shape, t)
        else:
            self._latest = self._make_packet(current_shape, current_shape, 0.0)

    def _make_packet(
        self,
        src: _HandShape,
        dst: _HandShape,
        alpha: float,
    ) -> GloveData:
        """Interpolate between two hand shapes and add noise."""
        self._sequence += 1

        # Smooth-step interpolation
        t = alpha * alpha * (3 - 2 * alpha)

        def lerp(a: float, b: float) -> float:
            return a + (b - a) * t

        flex = [
            float(
                max(0.0, min(1.0, lerp(s, d) + random.gauss(0, self.FLEX_NOISE)))
            )
            for s, d in zip(src.flex, dst.flex)
        ]

        # Orientation (Euler in degrees → radians → quaternion)
        roll = _deg2rad(lerp(src.orientation_deg[0], dst.orientation_deg[0]))
        pitch = _deg2rad(lerp(src.orientation_deg[1], dst.orientation_deg[1]))
        yaw = _deg2rad(lerp(src.orientation_deg[2], dst.orientation_deg[2]))

        # Simulated accelerometer: gravity component + noise
        accel = [
            random.gauss(0, self.ACCEL_NOISE),
            random.gauss(0, self.ACCEL_NOISE),
            float(9.81 * math.cos(pitch) + random.gauss(0, self.ACCEL_NOISE)),
        ]

        # Simulated gyro: low-frequency drift + noise
        slow = 2 * math.pi * 0.5 * time.time()
        gyro = [
            float(0.02 * math.sin(slow) + random.gauss(0, self.GYRO_NOISE)),
            float(0.02 * math.cos(slow) + random.gauss(0, self.GYRO_NOISE)),
            random.gauss(0, self.GYRO_NOISE),
        ]

        quat = _euler_to_quat(roll, pitch, yaw)

        # BLE quality: simulate occasional poor connection
        ble_q = max(0.0, min(1.0, 0.95 + random.gauss(0, 0.03)))

        return GloveData(
            flex=flex,
            imu=IMUData(accel=accel, gyro=gyro, quaternion=quat),
            ble_quality=ble_q,
            sequence=self._sequence,
            is_mock=True,
        )


# ───────────────── real BLE stub (future hardware) ──────────────


class BLEGloveSensor(GloveSensorInterface):
    """
    Real ESP32 BLE glove interface.

    TO IMPLEMENT when hardware is ready:
    1. Scan for the ESP32 advertising "HandTalk-Glove" service UUID.
    2. Connect with bleak (async BLE library for Python).
    3. Subscribe to the GATT notify characteristic for sensor packets.
    4. Parse the binary frame: [seq:uint16, flex×5:uint8, accel×3:int16,
                                gyro×3:int16, quat×4:int16] (little-endian).
    5. Convert raw ADC/fixed-point values to float using the calibration map.
    6. Compute ble_quality from RSSI and packet-loss counter.
    """

    BLE_SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
    BLE_CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

    def __init__(self) -> None:
        # Calibration offsets loaded from disk (produced by calibration routine)
        self._flex_min = [0] * 5
        self._flex_max = [1023] * 5
        self._latest: GloveData | None = None
        self._client = None

    async def start(self) -> None:
        raise NotImplementedError(
            "BLEGloveSensor.start() — connect real hardware first.\n"
            "Set USE_MOCK_GLOVE=false in .env and implement this method."
        )

    async def stop(self) -> None:
        if self._client:
            await self._client.disconnect()

    async def read(self) -> GloveData:
        if self._latest is None:
            raise RuntimeError("BLE glove not connected")
        return self._latest
