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
      minDetectionConfidence: 0.3,  // 낮출수록 장갑 착용 시에도 감지됨
      minTrackingConfidence: 0.3,
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

    let right = null;
    let left  = null;

    if (results.multiHandLandmarks?.length > 0) {
      results.multiHandLandmarks.forEach((lms, i) => {
        const label = results.multiHandedness?.[i]?.label ?? 'Right';
        const color = label === 'Right' ? '#7c6af7' : '#f87171';
        drawConnectors(this.#ctx, lms, HAND_CONNECTIONS, { color, lineWidth: 2 });
        drawLandmarks(this.#ctx, lms, { color, lineWidth: 1, radius: 3 });

        const wlms = results.multiHandWorldLandmarks?.[i] ?? lms;
        const conf = results.multiHandedness?.[i]?.score ?? 0.9;
        const data = {
          landmarks:       lms.map(p => ({ x: p.x, y: p.y, z: p.z })),
          world_landmarks: wlms.map(p => ({ x: p.x, y: p.y, z: p.z })),
          confidence:      conf,
          handedness:      label,
          fps:             30,
        };
        if (label === 'Right') right = data;
        else                   left  = data;
      });
    }

    this.#ctx.restore();
    this.#callback?.({ right, left });
  }
}
