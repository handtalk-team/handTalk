/**
 * CameraCapture — 웹캠 + MediaPipe Hands (index.html 의 CDN 스크립트).
 * 서버에는 영상 대신 손 랜드마크(21×3)만 보낸다. 서버 SensorFrame.camera 와 같은 모양.
 * 장갑을 끼면 MediaPipe 가 손을 못 찾으므로 GloveRecolor 로 모델 입력만 피부색으로 바꾼다.
 */
import { GloveRecolor } from './glove_recolor.js';

export const RECOLOR_KEY = 'handtalk.gloveRecolor';

export function saveRecolor(settings) {
  try { localStorage.setItem(RECOLOR_KEY, JSON.stringify(settings)); } catch { /* 저장 불가 */ }
}

export class CameraCapture {
  #video; #canvas; #ctx; #hands = null; #mpCamera = null; #stream = null;
  #callback = null;
  drawLandmarks = true;
  recolor = new GloveRecolor();

  constructor(videoEl, canvasEl) {
    this.#video = videoEl;
    this.#canvas = canvasEl;
    this.#ctx = canvasEl.getContext('2d');
    try { Object.assign(this.recolor.settings, JSON.parse(localStorage.getItem(RECOLOR_KEY) || '{}')); } catch { /* 기본값 */ }
  }

  /** 화면 위 정규화 좌표(0~1, 좌우 반전 전)의 장갑 색 */
  pickColor(nx, ny) { return this.recolor.sampleColor(this.#video, nx, ny); }

  onFrame(fn) { this.#callback = fn; }

  get running() { return !!this.#stream; }

  async start() {
    if (this.#stream) return;
    if (typeof Hands === 'undefined') throw new Error('MediaPipe 를 불러오지 못했습니다 (인터넷 연결 확인)');
    this.#stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, frameRate: 30 }, audio: false });
    this.#video.srcObject = this.#stream;
    await new Promise((r) => { this.#video.onloadedmetadata = r; });
    await this.#video.play();
    this.#canvas.width = this.#video.videoWidth;
    this.#canvas.height = this.#video.videoHeight;

    this.#hands = new Hands({ locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}` });
    this.#hands.setOptions({ maxNumHands: 2, modelComplexity: 1, minDetectionConfidence: 0.3, minTrackingConfidence: 0.3 });
    this.#hands.onResults((r) => this.#onResults(r));
    this.#mpCamera = new Camera(this.#video, {
      onFrame: async () => {
        if (!this.#hands) return;
        const image = this.recolor.active ? this.recolor.process(this.#video) : this.#video;
        await this.#hands.send({ image });
      },
      width: 640, height: 480,
    });
    this.#mpCamera.start();
  }

  stop() {
    this.#mpCamera?.stop();
    this.#stream?.getTracks().forEach((t) => t.stop());
    this.#stream = null;
    this.#hands?.close?.();
    this.#hands = null;
    this.#ctx.clearRect(0, 0, this.#canvas.width, this.#canvas.height);
  }

  #onResults(results) {
    const ctx = this.#ctx;
    ctx.clearRect(0, 0, this.#canvas.width, this.#canvas.height);
    // 보정 화면 보기: 모델이 실제로 받은 영상을 겹쳐 그림
    if (this.recolor.settings.preview && results.image) ctx.drawImage(results.image, 0, 0, this.#canvas.width, this.#canvas.height);
    let right = null;
    let left = null;
    (results.multiHandLandmarks || []).forEach((lms, i) => {
      const label = results.multiHandedness?.[i]?.label ?? 'Right';
      if (this.drawLandmarks) {
        const color = label === 'Right' ? '#6fcbbf' : '#f6a26b';
        drawConnectors(ctx, lms, HAND_CONNECTIONS, { color, lineWidth: 3 });
        drawLandmarks(ctx, lms, { color: '#ffffff', fillColor: color, lineWidth: 1, radius: 3 });
      }
      const wlms = results.multiHandWorldLandmarks?.[i] ?? lms;
      const data = {
        landmarks: lms.map((p) => ({ x: p.x, y: p.y, z: p.z })),
        world_landmarks: wlms.map((p) => ({ x: p.x, y: p.y, z: p.z })),
        confidence: results.multiHandedness?.[i]?.score ?? 0.9,
        handedness: label,
        fps: 30,
      };
      if (label === 'Right') right = data; else left = data;
    });
    this.#callback?.({ right, left });
  }
}
