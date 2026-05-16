/**
 * GloveMock — browser-side mock glove sensor
 *
 * Mirrors the server-side MockGloveSensor in Python.
 * Produces the exact same JSON shape as a real ESP32 BLE packet would.
 *
 * To switch to real BLE:
 *   1. Implement BLEGlove (same interface as GloveMock)
 *   2. Replace `new GloveMock()` with `new BLEGlove()` in app.js
 *   3. Nothing else needs to change.
 *
 * Packet shape:
 * {
 *   flex:        [thumb, index, middle, ring, pinky],   // 0.0–1.0
 *   imu: {
 *     accel:     [x, y, z],     // m/s²
 *     gyro:      [x, y, z],     // rad/s
 *     quaternion:[w, x, y, z],
 *   },
 *   ble_quality: 0.0–1.0,
 *   sequence:    int,
 *   is_mock:     true,
 * }
 */

const HAND_SHAPES = [
  { name: 'open',     flex: [0.05, 0.05, 0.05, 0.05, 0.05], orient: [0,  0,  0]  },
  { name: 'fist',     flex: [0.95, 0.95, 0.95, 0.95, 0.95], orient: [0,  0,  0]  },
  { name: 'point',    flex: [0.90, 0.05, 0.90, 0.90, 0.90], orient: [0,  5, 10]  },
  { name: 'peace',    flex: [0.90, 0.05, 0.05, 0.90, 0.90], orient: [0,  5,  5]  },
  { name: 'ok',       flex: [0.70, 0.70, 0.10, 0.10, 0.10], orient: [10,-20,  5] },
  { name: 'thumbUp',  flex: [0.05, 0.90, 0.90, 0.90, 0.90], orient: [0, 30,  0]  },
  { name: 'pinch',    flex: [0.60, 0.60, 0.10, 0.10, 0.10], orient: [0,  0,  0]  },
  { name: 'wave',     flex: [0.10, 0.10, 0.10, 0.10, 0.10], orient: [0, 15, 30]  },
  { name: 'cup',      flex: [0.40, 0.50, 0.50, 0.50, 0.50], orient: [0,  0,  0]  },
  { name: 'lShape',   flex: [0.05, 0.05, 0.90, 0.90, 0.90], orient: [0,-10,  0]  },
];

const HOLD_MS = 1500;
const TRANS_MS = 500;
const FLEX_NOISE = 0.015;
const ACCEL_NOISE = 0.08;
const GYRO_NOISE = 0.005;

function gauss(std = 1) {
  // Box-Muller
  const u = 1 - Math.random();
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * std;
}

function clamp(v, lo = 0, hi = 1) { return Math.max(lo, Math.min(hi, v)); }

function deg2rad(d) { return d * Math.PI / 180; }

function eulerToQuat(rollDeg, pitchDeg, yawDeg) {
  const r = deg2rad(rollDeg) / 2;
  const p = deg2rad(pitchDeg) / 2;
  const y = deg2rad(yawDeg) / 2;
  const cr = Math.cos(r), sr = Math.sin(r);
  const cp = Math.cos(p), sp = Math.sin(p);
  const cy = Math.cos(y), sy = Math.sin(y);
  return [
    cr * cp * cy + sr * sp * sy,
    sr * cp * cy - cr * sp * sy,
    cr * sp * cy + sr * cp * sy,
    cr * cp * sy - sr * sp * cy,
  ];
}

function smoothStep(t) { return t * t * (3 - 2 * t); }

export class GloveMock {
  #seq = 0;
  #shapeIdx = 0;
  #phaseStart = performance.now();
  #inTransition = false;
  #latest = null;
  #intervalId = null;

  start(hz = 50) {
    this.#tick();
    this.#intervalId = setInterval(() => this.#tick(), 1000 / hz);
  }

  stop() {
    if (this.#intervalId !== null) {
      clearInterval(this.#intervalId);
      this.#intervalId = null;
    }
  }

  /** Returns the latest sensor packet (non-blocking). */
  read() {
    return this.#latest;
  }

  #tick() {
    const now = performance.now();
    const elapsed = now - this.#phaseStart;

    const src = HAND_SHAPES[this.#shapeIdx];
    const dst = HAND_SHAPES[(this.#shapeIdx + 1) % HAND_SHAPES.length];

    let alpha = 0;

    if (!this.#inTransition) {
      if (elapsed >= HOLD_MS) {
        this.#inTransition = true;
        this.#phaseStart = now;
      }
    } else {
      alpha = Math.min(elapsed / TRANS_MS, 1);
      if (alpha >= 1) {
        this.#shapeIdx = (this.#shapeIdx + 1) % HAND_SHAPES.length;
        this.#inTransition = false;
        this.#phaseStart = now;
        alpha = 1;
      }
    }

    const t = smoothStep(alpha);

    const flex = src.flex.map((s, i) =>
      clamp(s + (dst.flex[i] - s) * t + gauss(FLEX_NOISE))
    );

    const roll  = src.orient[0] + (dst.orient[0] - src.orient[0]) * t;
    const pitch = src.orient[1] + (dst.orient[1] - src.orient[1]) * t;
    const yaw   = src.orient[2] + (dst.orient[2] - src.orient[2]) * t;
    const quat  = eulerToQuat(roll, pitch, yaw);

    const pitchRad = deg2rad(pitch);
    const accel = [
      gauss(ACCEL_NOISE),
      gauss(ACCEL_NOISE),
      9.81 * Math.cos(pitchRad) + gauss(ACCEL_NOISE),
    ];

    const slow = 2 * Math.PI * 0.5 * (Date.now() / 1000);
    const gyro = [
      0.02 * Math.sin(slow) + gauss(GYRO_NOISE),
      0.02 * Math.cos(slow) + gauss(GYRO_NOISE),
      gauss(GYRO_NOISE),
    ];

    this.#latest = {
      flex,
      imu: { accel, gyro, quaternion: quat },
      ble_quality: clamp(0.95 + gauss(0.03)),
      sequence: ++this.#seq,
      is_mock: true,
    };
  }
}

// ─── Stub for real BLE (implement when hardware is ready) ────────
export class BLEGlove {
  #device = null;
  #latest = null;
  #seq = 0;

  static SERVICE_UUID = '4fafc201-1fb5-459e-8fcc-c5c9c331914b';
  static CHAR_UUID    = 'beb5483e-36e1-4688-b7f5-ea07361b26a8';

  async start() {
    this.#device = await navigator.bluetooth.requestDevice({
      filters: [{ name: 'HandTalk-Glove' }],
      optionalServices: [BLEGlove.SERVICE_UUID],
    });
    const server = await this.#device.gatt.connect();
    const service = await server.getPrimaryService(BLEGlove.SERVICE_UUID);
    const char = await service.getCharacteristic(BLEGlove.CHAR_UUID);
    await char.startNotifications();
    char.addEventListener('characteristicvaluechanged', (e) => {
      this.#latest = this.#parse(e.target.value);
    });
    console.log('BLE Glove connected:', this.#device.name);
  }

  stop() {
    this.#device?.gatt.disconnect();
  }

  read() { return this.#latest; }

  /** Parse binary BLE frame from ESP32.
   *  Layout: [seq:u16, flex×5:u8, ax:i16, ay:i16, az:i16,
   *           gx:i16, gy:i16, gz:i16, qw:i16, qx:i16, qy:i16, qz:i16]
   *  All little-endian.  Fixed-point: flex /255, IMU /1000.
   */
  #parse(dv) {
    const seq  = dv.getUint16(0, true);
    const flex = Array.from({ length: 5 }, (_, i) => dv.getUint8(2 + i) / 255);
    const off  = 7;
    const accel = [0, 1, 2].map(i => dv.getInt16(off + i * 2, true) / 1000);
    const gyro  = [0, 1, 2].map(i => dv.getInt16(off + 6 + i * 2, true) / 1000);
    const quat  = [0, 1, 2, 3].map(i => dv.getInt16(off + 12 + i * 2, true) / 10000);
    this.#latest = {
      flex, imu: { accel, gyro, quaternion: quat },
      ble_quality: 1.0,
      sequence: seq,
      is_mock: false,
    };
    return this.#latest;
  }
}
