/**
 * BLEGlove — real ESP32 BLE glove stub.
 *
 * To enable when hardware is ready:
 *   1. Replace `glove = null` with `glove = new BLEGlove()` in app.js
 *   2. Call `await glove.start()` in the boot block
 *   3. Pass `glove.read()` as the glove field in each frame
 *
 * Packet layout (28 bytes, little-endian):
 *   [seq:u16, flex×5:u8, ax:i16, ay:i16, az:i16,
 *    gx:i16, gy:i16, gz:i16, qw:i16, qx:i16, qy:i16, qz:i16, quality:u8]
 *   Fixed-point: flex /255, IMU /1000, quat /10000
 */
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
    const server  = await this.#device.gatt.connect();
    const service = await server.getPrimaryService(BLEGlove.SERVICE_UUID);
    const char    = await service.getCharacteristic(BLEGlove.CHAR_UUID);
    await char.startNotifications();
    char.addEventListener('characteristicvaluechanged', (e) => {
      this.#latest = this.#parse(e.target.value);
    });
    console.log('BLE Glove connected:', this.#device.name);
  }

  stop() { this.#device?.gatt.disconnect(); }

  read() { return this.#latest; }

  #parse(dv) {
    const seq  = dv.getUint16(0, true);
    const flex = Array.from({ length: 5 }, (_, i) => dv.getUint8(2 + i) / 255);
    const accel = [0, 1, 2].map(i => dv.getInt16(7 + i * 2, true) / 1000);
    const gyro  = [0, 1, 2].map(i => dv.getInt16(13 + i * 2, true) / 1000);
    const quat  = [0, 1, 2, 3].map(i => dv.getInt16(19 + i * 2, true) / 10000);
    const quality = dv.getUint8(27) / 255;
    return {
      flex,
      imu: { accel, gyro, quaternion: quat },
      ble_quality: quality,
      sequence: seq,
      is_mock: false,
    };
  }
}
