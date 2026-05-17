"""
Glove Sensor Interface
======================

BLEGloveSensor — stub for the real ESP32 glove.

TO IMPLEMENT when hardware is ready:
1. Install bleak:  pip install bleak
2. Implement start() to scan & connect via BLE.
3. Parse the 28-byte binary GATT notification:
     [seq:uint16, flex×5:uint8, accel×3:int16,
      gyro×3:int16, quat×4:int16, quality:uint8]  (little-endian)
4. Set USE_MOCK_GLOVE=false in .env — nothing else needs to change.

When no glove is connected, SensorFrame.glove = None and the fusion
module automatically falls back to vision-only mode (glove weight = 0).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.schemas.sensor import GloveData


class GloveSensorInterface(ABC):
    @abstractmethod
    async def read(self) -> GloveData | None: ...

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...


class BLEGloveSensor(GloveSensorInterface):
    """Real ESP32 BLE glove — implement when hardware is ready."""

    BLE_SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
    BLE_CHAR_UUID    = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

    def __init__(self) -> None:
        self._latest: GloveData | None = None
        self._client = None

    async def start(self) -> None:
        # TODO: bleak BLE scan & connect
        pass

    async def stop(self) -> None:
        if self._client:
            await self._client.disconnect()

    async def read(self) -> GloveData | None:
        return self._latest  # None until BLE connects
