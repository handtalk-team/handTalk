/**
 * CameraCapture
 * =============
 * Accesses the laptop webcam via getUserMedia, runs MediaPipe Hands
 * in the browser, and calls the registered callback with VisionData
 * (same shape expected by the server's SensorFrame.camera field).
 *
 * MediaPipe is loaded from CDN via index.html script tags.
 * Landmarks are drawn onto an overlay <canvas> for visual feedback.
 *
 * Production notes:
 *  - Running MediaPipe in the browser means the server receives only
 *    compact JSON (21 × 3 floats ≈ 500 bytes/frame) instead of
 *    raw video (~100 KB/frame) — critical for the 2-second latency budget.
 *  - If the browser can't handle real-time MediaPipe (low-end device),
 *    switch to sending compressed JPEG frames and run MediaPipe server-side.
 */

export class CameraCapture {
  #video;
  #canvas;
  #ctx;
  #callback = null;
  #hands = null;
  #mpCamera = null;

  constructor(videoEl, canvasEl) {
    this.#video  = videoEl;
    this.#canvas = canvasEl;
    this.#ctx    = canvasEl.getContext('2d');
  }

  /** Register a callback: fn(visionData | null) called at ~30 Hz. */
  onFrame(fn) {
    this.#callback = fn;
  }

  async start() {
    // Request webcam
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, frameRate: 30 },
      audio: false,
    });
    this.#video.srcObject = stream;
    await new Promise(r => this.#video.onloadedmetadata = r);
    this.#video.play();

    // Sync canvas size
    this.#canvas.width  = this.#video.videoWidth;
    this.#canvas.height = this.#video.videoHeight;

    // Initialise MediaPipe Hands
    this.#hands = new Hands({
      locateFile: (f) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`,
    });
    this.#hands.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    this.#hands.onResults((results) => this.#onResults(results));

    // MediaPipe camera loop
    this.#mpCamera = new Camera(this.#video, {
      onFrame: async () => {
        await this.#hands.send({ image: this.#video });
      },
      width: 640,
      height: 480,
    });
    this.#mpCamera.start();
  }

  stop() {
    this.#mpCamera?.stop();
    const stream = this.#video.srcObject;
    stream?.getTracks().forEach(t => t.stop());
  }

  // ── MediaPipe results handler ─────────────────────────────────

  #onResults(results) {
    this.#ctx.save();
    this.#ctx.clearRect(0, 0, this.#canvas.width, this.#canvas.height);

    let visionData = null;

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      // Draw all detected hands
      results.multiHandLandmarks.forEach((lms, i) => {
        const color = i === 0 ? '#7c6af7' : '#f59e0b';
        drawConnectors(this.#ctx, lms, HAND_CONNECTIONS, { color, lineWidth: 2 });
        drawLandmarks(this.#ctx, lms, { color, lineWidth: 1, radius: 3 });
      });

      // Send the most confident hand to the server (or right hand preferred)
      const idx = results.multiHandedness?.findIndex(h => h.label === 'Right') ?? 0;
      const best = Math.max(0, idx);
      const lms  = results.multiHandLandmarks[best];
      const wlms = results.multiHandWorldLandmarks?.[best] ?? lms;
      const hand = results.multiHandedness?.[best]?.label ?? 'Right';
      const conf = results.multiHandedness?.[best]?.score ?? 0.9;

      visionData = {
        landmarks:       lms.map(p => ({ x: p.x, y: p.y, z: p.z })),
        world_landmarks: wlms.map(p => ({ x: p.x, y: p.y, z: p.z })),
        confidence:      conf,
        handedness:      hand,
        fps:             30,
      };
    }

    this.#ctx.restore();
    this.#callback?.(visionData);
  }
}
