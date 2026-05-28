"""
Glove Sensor Interface
======================

SerialGloveSensor — ESP32 USB Serial glove (CSV format, 20 Hz).

ESP32 output format (one line per packet):
    R,flex1,flex2,flex3,flex4,flex5,ax,ay,az,gx,gy,gz

Field details:
  Side         : "R" (right) or "L" (left)
  flex1-5      : raw ADC (0–4095), thumb→pinky
  ax,ay,az     : raw MPU-9250 accel (int16, ±2 g range)
  gx,gy,gz     : raw MPU-9250 gyro  (int16, ±250 °/s range)

Normalisation applied here:
  flex  : (raw - 800) / 500, clipped to [0, 1]
  accel : raw / 16384 * 9.81  → m/s²
  gyro  : raw / 131 * π/180   → rad/s

Configuration (.env):
  GLOVE_PORT_RIGHT=/dev/cu.usbserial-0001   # macOS / Linux
  GLOVE_PORT_LEFT=                          # leave empty if single glove
  GLOVE_BAUD_RATE=115200                    # must match ESP32 Serial.begin()
"""

from __future__ import annotations

import math
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

import serial  # pyserial

from app.models.schemas.sensor import GloveData, IMUData

# ── normalisation constants ────────────────────────────────────────────────────
_FLEX_ZERO    = 800     # ADC value when finger is fully open
_FLEX_SCALE   = 500     # ADC range from open to closed

_ACCEL_SCALE  = 16384.0 # LSB/g  (MPU-9250, ±2 g mode)
_G            = 9.81    # m/s²

_GYRO_SCALE   = 131.0   # LSB/(°/s) (MPU-9250, ±250 °/s mode)
_DEG2RAD      = math.pi / 180.0

_QUAT_IDENTITY = [1.0, 0.0, 0.0, 0.0]  # ESP32 does not send quaternion yet


def _parse_line(line: str) -> Optional[tuple[str, GloveData]]:
    """
    Parse one CSV line from the ESP32.
    Returns (side, GloveData) or None if the line is malformed.
    """
    parts = line.strip().split(",")
    if len(parts) != 12:
        return None

    try:
        side = parts[0].strip().upper()
        if side not in ("R", "L"):
            return None

        flex_raw = [float(p) for p in parts[1:6]]
        ax, ay, az = (float(p) for p in parts[6:9])
        gx, gy, gz = (float(p) for p in parts[9:12])
    except ValueError:
        return None

    flex_norm = [
        float(max(0.0, min(1.0, (v - _FLEX_ZERO) / _FLEX_SCALE)))
        for v in flex_raw
    ]

    accel = [ax / _ACCEL_SCALE * _G,
             ay / _ACCEL_SCALE * _G,
             az / _ACCEL_SCALE * _G]
    gyro  = [gx / _GYRO_SCALE * _DEG2RAD,
             gy / _GYRO_SCALE * _DEG2RAD,
             gz / _GYRO_SCALE * _DEG2RAD]

    data = GloveData(
        flex=flex_norm,
        imu=IMUData(accel=accel, gyro=gyro, quaternion=_QUAT_IDENTITY),
        ble_quality=1.0,
        is_mock=False,
    )
    return side, data


# ── Abstract interface ─────────────────────────────────────────────────────────

class GloveSensorInterface(ABC):
    @abstractmethod
    async def read(self) -> GloveData | None: ...

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...


# ── Serial implementation ──────────────────────────────────────────────────────

class SerialGloveSensor(GloveSensorInterface):
    """
    Reads one ESP32 glove over USB serial in a background thread.

    Parameters
    ----------
    port      : serial port path, e.g. "/dev/cu.usbserial-0001"
    baud_rate : must match ESP32 Serial.begin() — default 115200
    side      : "R" or "L" — only packets matching this side are kept
    """

    def __init__(self, port: str, baud_rate: int = 115200, side: str = "R") -> None:
        self._port      = port
        self._baud_rate = baud_rate
        self._side      = side.upper()
        self._latest: Optional[GloveData] = None
        self._running   = False
        self._thread: Optional[threading.Thread] = None
        self._ser: Optional[serial.Serial] = None

    async def start(self) -> None:
        if self._running:
            return
        try:
            self._ser = serial.Serial(self._port, self._baud_rate, timeout=1.0)
        except serial.SerialException as exc:
            print(f"[GloveSensor] Cannot open {self._port}: {exc}")
            return

        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print(f"[GloveSensor] Listening on {self._port} @ {self._baud_rate} baud (side={self._side})")

    async def stop(self) -> None:
        self._running = False
        if self._ser and self._ser.is_open:
            self._ser.close()

    async def read(self) -> GloveData | None:
        return self._latest

    # ── background thread ──────────────────────────────────────────────────────

    def _read_loop(self) -> None:
        seq = 0
        while self._running and self._ser and self._ser.is_open:
            try:
                raw = self._ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore")
                result = _parse_line(line)
                if result is None:
                    continue
                side, data = result
                if side != self._side:
                    continue
                data.sequence = seq
                seq += 1
                self._latest = data
            except serial.SerialException:
                print(f"[GloveSensor] Serial error on {self._port} — stopping reader")
                break
            except Exception as exc:
                print(f"[GloveSensor] Unexpected error: {exc}")
                time.sleep(0.01)


# ── Legacy BLE stub (kept for reference) ──────────────────────────────────────

class BLEGloveSensor(GloveSensorInterface):
    """Placeholder — not implemented.  Use SerialGloveSensor instead."""

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def read(self) -> GloveData | None:
        return None
