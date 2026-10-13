/**
 * GloveRecolor
 * ============
 * MediaPipe Hands는 맨손 사진으로 학습돼 있어서, 글러브를 낀 손은
 * "손"으로 잘 검출하지 못합니다. 재학습 없이 이를 보완하려고 MediaPipe에
 * 넣기 직전 프레임에서 글러브 픽셀을 피부색으로 바꿔 줍니다.
 *
 *  - 빨간 장갑 자동: 채도 높은 빨간 픽셀(팀 센서 글러브 원단)을 자동으로 피부색으로.
 *  - 글러브 색 찍기: 영상에서 글러브를 클릭해 기준 색을 추가로 지정 (다른 색 장갑용).
 *  - 검은 손끝 보정: 글러브 바로 옆의 어두운 픽셀(검은 손가락 끝 고무)도 피부색으로.
 *  - 원래 밝기 차이는 유지해서 손가락 마디 음영이 남도록 합니다.
 *
 * 화면에 보이는 영상은 그대로이고, 모델 입력만 바뀝니다.
 */

const SKIN_Y  = 170;   // 바꿀 피부색의 기준 밝기
const SKIN_CB = 108;
const SKIN_CR = 150;
const CELL    = 4;     // 손끝 탐색용 저해상도 격자 크기 (px)

export const DEFAULT_SETTINGS = {
  enabled: false,
  autoRed: true,       // 빨간 장갑 자동 검출
  darkTips: true,      // 글러브 옆 검은 손끝/코팅도 보정
  darkMax: 90,         // 이 밝기보다 어두운 무채색을 '검은 글러브'로 봄
  tipReach: 60,        // 글러브 옆 검은 부분을 찾을 거리 (px, 640 폭 기준)
  color: null,         // [r, g, b] — 직접 찍은 글러브 색 (선택)
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

function writeSkin(d, i, y) {
  const ny = Math.max(0, Math.min(255, y));
  const cbo = SKIN_CB - 128, cro = SKIN_CR - 128;
  d[i]     = ny + 1.402 * cro;
  d[i + 1] = ny - 0.344136 * cbo - 0.714136 * cro;
  d[i + 2] = ny + 1.772 * cbo;
}

export class GloveRecolor {
  #canvas = document.createElement('canvas');
  #ctx    = this.#canvas.getContext('2d', { willReadFrequently: true });
  settings = { ...DEFAULT_SETTINGS };

  /** 보정을 거쳐야 하는지 (아니면 원본 영상을 그대로 MediaPipe에 넘김). */
  get active() {
    const s = this.settings;
    return (s.enabled && (s.autoRed || s.color)) || s.brightness !== 100 || s.contrast !== 100;
  }

  /** video(또는 img/canvas) 프레임을 보정한 canvas를 반환합니다. */
  process(src) {
    const w = src.videoWidth || src.naturalWidth || src.width;
    const h = src.videoHeight || src.naturalHeight || src.height;
    if (!w || !h) return src;
    if (this.#canvas.width !== w)  this.#canvas.width = w;
    if (this.#canvas.height !== h) this.#canvas.height = h;

    const s = this.settings;
    this.#ctx.filter = `brightness(${s.brightness}%) contrast(${s.contrast}%)`;
    this.#ctx.drawImage(src, 0, 0, w, h);
    this.#ctx.filter = 'none';

    if (s.enabled && (s.autoRed || s.color)) this.#recolor(w, h);
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
    const ref = s.color ? toYCbCr(...s.color) : null;
    const tol2 = s.tolerance * s.tolerance;
    // 밝기는 색차보다 조명 영향을 많이 받으므로 넉넉하게 허용
    const yTol = 60 + s.tolerance * 2;

    const img = this.#ctx.getImageData(0, 0, w, h);
    const d = img.data;
    const n = w * h;
    const glove = new Uint8Array(n);

    // 1) 글러브 원단 픽셀 → 피부색
    for (let p = 0, i = 0; p !== n; p++, i += 4) {
      const r = d[i], g = d[i + 1], b = d[i + 2];
      const y  = 0.299 * r + 0.587 * g + 0.114 * b;
      const cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
      const cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;

      let hit = false;
      if (s.autoRed) {
        // 원단의 진한 빨강 (피부보다 훨씬 채도 높음 — 피부가 걸려도 피부색으로 바뀔 뿐)
        hit = cr > 155 && cr - cb > 45 && r > 1.6 * g && r > 1.4 * b && r > 70;
      }
      if (!hit && ref && Math.abs(y - ref[0]) <= yTol) {
        const dcb = cb - ref[1], dcr = cr - ref[2];
        hit = dcb * dcb + dcr * dcr <= tol2;
      }
      if (!hit) continue;
      glove[p] = 1;
      const refY = ref ? ref[0] : 90;   // 빨간 원단 평균 밝기 ≈ 90
      writeSkin(d, i, SKIN_Y + (y - refY) * 0.6);
    }

    // 2) 글러브 옆 어두운 픽셀(검은 손끝 고무) → 조금 어두운 피부색
    if (s.darkTips) {
      const gw = Math.ceil(w / CELL), gh = Math.ceil(h / CELL);
      // 격자별 글러브 픽셀 수 → 적분 영상으로 반경 내 글러브 존재 여부를 빠르게 계산
      const cnt = new Uint32Array((gw + 1) * (gh + 1));
      for (let y = 0; y !== h; y++) {
        const row = y * w, gy = (y / CELL) | 0;
        for (let x = 0; x !== w; x++) {
          if (glove[row + x]) cnt[(gy + 1) * (gw + 1) + ((x / CELL) | 0) + 1]++;
        }
      }
      for (let gy = 1; gy <= gh; gy++) {
        let acc = 0;
        for (let gx = 1; gx <= gw; gx++) {
          acc += cnt[gy * (gw + 1) + gx];
          cnt[gy * (gw + 1) + gx] = acc + cnt[(gy - 1) * (gw + 1) + gx];
        }
      }
      const rad = Math.max(1, Math.round((s.tipReach * (w / 640)) / CELL));
      const near = new Uint8Array(gw * gh);
      for (let gy = 0; gy !== gh; gy++) {
        const y0 = Math.max(0, gy - rad), y1 = Math.min(gh, gy + rad + 1);
        for (let gx = 0; gx !== gw; gx++) {
          const x0 = Math.max(0, gx - rad), x1 = Math.min(gw, gx + rad + 1);
          const sum = cnt[y1 * (gw + 1) + x1] - cnt[y0 * (gw + 1) + x1]
                    - cnt[y1 * (gw + 1) + x0] + cnt[y0 * (gw + 1) + x0];
          // 주변에 글러브 픽셀이 충분히 있어야 (배경의 검은 물체는 제외)
          near[gy * gw + gx] = sum > CELL * CELL * 2 ? 1 : 0;
        }
      }
      for (let y = 0; y !== h; y++) {
        const row = y * w, grow = ((y / CELL) | 0) * gw;
        for (let x = 0; x !== w; x++) {
          const p = row + x;
          if (glove[p] || !near[grow + ((x / CELL) | 0)]) continue;
          const i = p * 4;
          const r = d[i], g = d[i + 1], b = d[i + 2];
          const y8 = 0.299 * r + 0.587 * g + 0.114 * b;
          const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
          if (y8 < s.darkMax && mx - mn < 45) writeSkin(d, i, SKIN_Y - 35 + y8 * 0.5);
        }
      }
    }
    this.#ctx.putImageData(img, 0, 0);
  }
}
