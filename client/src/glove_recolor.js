/**
 * GloveRecolor
 * ============
 * MediaPipe Hands는 맨손 사진으로 학습돼 있어서, 단색 글러브를 낀 손은
 * "손"으로 잘 검출하지 못합니다. 재학습 없이 이를 보완하려고 MediaPipe에
 * 넣기 직전 프레임에서 글러브 색 픽셀을 피부색으로 바꿔 줍니다.
 *
 *  - 사용자가 영상에서 글러브를 클릭해 기준 색을 고릅니다.
 *  - YCbCr 공간에서 색차(Cb, Cr)와 밝기(Y)가 기준 색에 가까운 픽셀만 바꿉니다.
 *  - 원래 밝기 차이는 유지해서 손가락 마디 음영이 남도록 합니다.
 *
 * 화면에 보이는 영상은 그대로이고, 모델 입력만 바뀝니다.
 */

const SKIN_Y  = 170;   // 바꿀 피부색의 기준 밝기
const SKIN_CB = 108;
const SKIN_CR = 150;

export const DEFAULT_SETTINGS = {
  enabled: false,
  color: null,         // [r, g, b] — 글러브 기준 색
  tolerance: 25,       // 색차 허용 범위 (Cb/Cr 거리)
  brightness: 100,     // %
  contrast: 100,       // %
  preview: false,      // 오버레이에 모델 입력 화면 표시
};

function toYCbCr(r, g, b) {
  return [
    0.299 * r + 0.587 * g + 0.114 * b,
    128 - 0.168736 * r - 0.331264 * g + 0.5 * b,
    128 + 0.5 * r - 0.418688 * g - 0.081312 * b,
  ];
}

export class GloveRecolor {
  #canvas = document.createElement('canvas');
  #ctx    = this.#canvas.getContext('2d', { willReadFrequently: true });
  settings = { ...DEFAULT_SETTINGS };

  /** 보정을 거쳐야 하는지 (아니면 원본 영상을 그대로 MediaPipe에 넘김). */
  get active() {
    const s = this.settings;
    return (s.enabled && s.color) || s.brightness !== 100 || s.contrast !== 100;
  }

  /** video 프레임을 보정한 canvas를 반환합니다. */
  process(video) {
    const w = video.videoWidth, h = video.videoHeight;
    if (!w || !h) return video;
    if (this.#canvas.width !== w)  this.#canvas.width = w;
    if (this.#canvas.height !== h) this.#canvas.height = h;

    const s = this.settings;
    this.#ctx.filter = `brightness(${s.brightness}%) contrast(${s.contrast}%)`;
    this.#ctx.drawImage(video, 0, 0, w, h);
    this.#ctx.filter = 'none';

    if (s.enabled && s.color) this.#recolor(w, h);
    return this.#canvas;
  }

  /** 정규화 좌표(0~1)의 주변 7×7 평균 색을 원본 영상에서 읽습니다. */
  sampleColor(video, nx, ny) {
    const w = video.videoWidth, h = video.videoHeight;
    this.#canvas.width = w;
    this.#canvas.height = h;
    this.#ctx.drawImage(video, 0, 0, w, h);
    const x = Math.max(0, Math.min(w - 7, Math.round(nx * w) - 3));
    const y = Math.max(0, Math.min(h - 7, Math.round(ny * h) - 3));
    const d = this.#ctx.getImageData(x, y, 7, 7).data;
    let r = 0, g = 0, b = 0;
    for (let i = 0; i != d.length; i += 4) { r += d[i]; g += d[i + 1]; b += d[i + 2]; }
    const n = d.length / 4;
    return [Math.round(r / n), Math.round(g / n), Math.round(b / n)];
  }

  #recolor(w, h) {
    const s = this.settings;
    const [refY, refCb, refCr] = toYCbCr(...s.color);
    const tol2 = s.tolerance * s.tolerance;
    // 밝기는 색차보다 조명 영향을 많이 받으므로 넉넉하게 허용
    const yTol = 60 + s.tolerance * 2;

    const img = this.#ctx.getImageData(0, 0, w, h);
    const d = img.data;
    for (let i = 0; i != d.length; i += 4) {
      const r = d[i], g = d[i + 1], b = d[i + 2];
      const y  = 0.299 * r + 0.587 * g + 0.114 * b;
      if (Math.abs(y - refY) > yTol) continue;
      const cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
      const cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;
      const dcb = cb - refCb, dcr = cr - refCr;
      if (dcb * dcb + dcr * dcr > tol2) continue;

      const ny = Math.max(0, Math.min(255, SKIN_Y + (y - refY)));
      const cbo = SKIN_CB - 128, cro = SKIN_CR - 128;
      d[i]     = ny + 1.402 * cro;
      d[i + 1] = ny - 0.344136 * cbo - 0.714136 * cro;
      d[i + 2] = ny + 1.772 * cbo;
    }
    this.#ctx.putImageData(img, 0, 0);
  }
}
