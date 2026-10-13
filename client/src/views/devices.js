/** 기기 점검 — 카메라 손 인식, 장갑 색 보정, 센서 장갑 값 확인 */
import { h, toast } from '../core/dom.js';
import { on } from '../core/store.js';
import { CameraCapture, saveRecolor } from '../core/camera.js';
import * as session from '../core/session.js';

const FINGERS = ['엄지', '검지', '중지', '약지', '새끼'];

export function render(root) {
  const video = h('video', { autoplay: true, muted: true, playsinline: true });
  const canvas = h('canvas');
  const handsChip = h('span.chip', '손 인식 대기');
  const fpsChip = h('span.chip.num', '0 fps');
  const camera = new CameraCapture(video, canvas);
  const rc = camera.recolor;
  let picking = false;

  const save = () => saveRecolor(rc.settings);
  const check = (key, label) => h('label.switch-row',
    h('div.t', label),
    h('input', { type: 'checkbox', checked: rc.settings[key], onchange: (e) => { rc.settings[key] = e.target.checked; save(); } }));
  const slider = (key, label, min, max, unit = '') => {
    const val = h('span.num.muted', `${rc.settings[key]}${unit}`);
    return h('div.field', { style: { padding: '8px 0' } },
      h('label', { style: { display: 'flex', justifyContent: 'space-between' } }, label, val),
      h('input', { type: 'range', min, max, value: rc.settings[key], oninput: (e) => { rc.settings[key] = Number(e.target.value); val.textContent = `${e.target.value}${unit}`; save(); } }));
  };
  const swatch = h('span', { style: { width: '22px', height: '22px', borderRadius: '6px', border: '1px solid var(--line-strong)', background: rc.settings.color ? `rgb(${rc.settings.color.join(',')})` : 'transparent' } });

  canvas.addEventListener('click', (e) => {
    if (!picking) return;
    picking = false;
    canvas.style.cursor = '';
    const r = canvas.getBoundingClientRect();
    // 화면은 거울처럼 좌우 반전되어 있으므로 x 를 되돌린다
    rc.settings.color = camera.pickColor(1 - (e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height);
    rc.settings.enabled = true;
    swatch.style.background = `rgb(${rc.settings.color.join(',')})`;
    save();
    toast(`장갑 색을 저장했어요: rgb(${rc.settings.color.join(', ')})`);
  });

  const gloveCols = { right: h('div'), left: h('div') };
  const drawGlove = (side, g) => {
    gloveCols[side].replaceChildren(
      h('div.eyebrow', side === 'right' ? '오른손' : '왼손'),
      g ? h('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '6px', marginTop: '8px' } },
        g.flex.map((v, i) => h('div', { style: { display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px', fontSize: '11px', color: 'var(--fg-3)' } },
          h('div', { style: { width: '100%', height: '64px', background: 'var(--surface-sunken)', borderRadius: '6px', position: 'relative', overflow: 'hidden' } },
            h('i', { style: { position: 'absolute', left: 0, right: 0, bottom: 0, height: `${Math.round(v * 100)}%`, background: 'linear-gradient(0deg, var(--brand-600), var(--brand-300))' } })),
          FINGERS[i])))
        : h('p.muted', { style: { fontSize: '13px', marginTop: '6px' } }, '연결되지 않음'));
  };
  drawGlove('right', null);
  drawGlove('left', null);

  const offs = [
    on('hands', ({ right, left }) => {
      const n = (right ? 1 : 0) + (left ? 1 : 0);
      handsChip.textContent = n ? `손 ${n}개 인식 중` : '손 인식 대기';
      handsChip.className = `chip ${n ? 'chip-ok' : ''}`;
    }),
    on('fps', (f) => { fpsChip.textContent = `${f} fps`; }),
    on('glove', (m) => { drawGlove('right', m.right); drawGlove('left', m.left); }),
  ];

  root.append(h('div.container',
    h('div.section-head', h('div', h('h2', { style: { fontSize: 'var(--fs-xl)' } }, '기기 점검'),
      h('p', '수업 전에 카메라와 장갑이 잘 인식되는지 확인하세요.'))),
    h('div', { style: { display: 'grid', gridTemplateColumns: 'minmax(0, 1.4fr) minmax(0, 1fr)', gap: '16px', alignItems: 'start' } },
      h('div.card', { style: { overflow: 'hidden' } },
        h('div.camera-wrap', { style: { aspectRatio: '4 / 3' } }, video, canvas, h('div.cam-hud', handsChip, fpsChip)),
        h('div.signal', h('span.muted', { style: { fontSize: '13px' } }, '손 전체가 화면에 들어오도록 앉고, 뒤쪽 배경은 단순할수록 좋아요.'))),
      h('div.card.card-pad',
        h('h2', { style: { fontSize: 'var(--fs-l)' } }, '장갑 색 보정'),
        h('p.muted', { style: { fontSize: '13px', margin: '4px 0 6px' } }, '장갑을 끼면 손 인식이 약해져요. 모델 입력에서만 장갑을 피부색으로 바꿔 인식을 돕습니다.'),
        check('enabled', '보정 켜기'),
        check('autoRed', '빨간 장갑 자동 인식'),
        check('darkTips', '검은 손끝·코팅 보정'),
        check('preview', '보정된 화면 보기'),
        h('div.switch-row', h('div.t', '장갑 색 직접 찍기'), h('div', { style: { display: 'flex', gap: '8px', alignItems: 'center' } }, swatch,
          h('button.btn.btn-sm', { onclick: () => { picking = true; canvas.style.cursor = 'crosshair'; toast('카메라 화면에서 장갑을 클릭하세요.'); } }, '찍기'))),
        slider('tolerance', '색 허용 범위', 5, 60),
        slider('tipReach', '검은 부분 범위', 10, 200, 'px'),
        slider('darkMax', '검은 기준 밝기', 40, 140),
        slider('brightness', '밝기', 50, 200, '%'),
        slider('contrast', '대비', 50, 200, '%'))),
    h('div.card.card-pad',
      h('div.section-head', h('h2', { style: { fontSize: 'var(--fs-l)' } }, '센서 장갑'), h('span.muted', { style: { fontSize: '13px' } }, '서버에 USB 로 연결된 ESP32 장갑의 굽힘 센서 값')),
      h('div', { style: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginTop: '12px' } }, gloveCols.left, gloveCols.right))));

  session.attachCamera(camera);
  camera.start().catch((e) => toast(`카메라를 켤 수 없어요: ${e.message || e.name}`, 'error'));
  return () => { offs.forEach((f) => f()); camera.stop(); };
}
