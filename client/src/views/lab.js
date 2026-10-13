/** 개발자 도구 — 인식 모델용 수어 샘플 녹화 (기존 데이터 수집 기능) */
import { h, toast } from '../core/dom.js';
import { on, state } from '../core/store.js';
import { CameraCapture } from '../core/camera.js';
import * as session from '../core/session.js';

export function render(root) {
  const video = h('video', { autoplay: true, muted: true, playsinline: true });
  const canvas = h('canvas');
  const label = h('input.input#label', { placeholder: '레이블 (예: 아프다)' });
  const counts = h('div.list');
  const bar = h('i', { style: { width: '0%' } });
  const camera = new CameraCapture(video, canvas);
  const tally = {};
  let busy = false;

  const record = () => {
    const l = label.value.trim();
    if (!l) return toast('레이블을 입력하세요.', 'warning');
    if (!session.isServerOnline()) return toast('서버에 연결되지 않았습니다.', 'error');
    if (busy) return;
    if (!state.session?.active) session.startSession({ mode: 'word', scenario: 'hospital', title: '데이터 수집' });
    busy = true;
    bar.style.transition = 'none'; bar.style.width = '0%';
    requestAnimationFrame(() => requestAnimationFrame(() => { bar.style.transition = 'width 2s linear'; bar.style.width = '100%'; }));
    session.collectSample(l);
    setTimeout(() => { busy = false; bar.style.width = '0%'; }, 2100);
  };

  const offs = [
    on('collect', (m) => {
      tally[m.label] = m.count;
      counts.replaceChildren(...Object.entries(tally).map(([k, v]) => h('div.list-row', h('div.grow.t', k), h('span.num', `${v}개`))));
    }),
  ];
  const keys = (e) => { if (e.code === 'Space' && document.activeElement !== label) { e.preventDefault(); record(); } };
  document.addEventListener('keydown', keys);

  root.append(h('div.container',
    h('div.section-head', h('div', h('h2', { style: { fontSize: 'var(--fs-xl)' } }, '데이터 수집'),
      h('p', '인식 모델 학습용 샘플을 2초씩 녹화해 서버에 저장해요. 개발용 기능입니다.'))),
    h('div', { style: { display: 'grid', gridTemplateColumns: 'minmax(0, 1.4fr) minmax(0, 1fr)', gap: '16px' } },
      h('div.card', { style: { overflow: 'hidden' } }, h('div.camera-wrap', { style: { aspectRatio: '4 / 3' } }, video, canvas)),
      h('div.card.card-pad', { style: { display: 'flex', flexDirection: 'column', gap: '12px' } },
        h('div.field', h('label', { for: 'label' }, '레이블'), label),
        h('button.btn.btn-primary.btn-lg', { onclick: record }, '녹화 시작 ', h('kbd', 'Space')),
        h('div.progress', bar),
        h('div.eyebrow', '저장된 샘플'), counts))));

  session.attachCamera(camera);
  camera.start().catch((e) => toast(`카메라를 켤 수 없어요: ${e.message}`, 'error'));
  return () => {
    offs.forEach((f) => f());
    document.removeEventListener('keydown', keys);
    camera.stop();
    if (state.session?.active) session.endSession({ save: false });
  };
}
