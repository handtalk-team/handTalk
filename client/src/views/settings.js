import { h, toast } from '../core/dom.js';
import { loadSettings, saveSettings, serverBase, state, on } from '../core/store.js';
import { loadCurriculum } from '../core/api.js';
import * as session from '../core/session.js';

export function render(root) {
  const s = loadSettings();
  const url = h('input.input#serverUrl', { value: s.serverUrl, placeholder: serverBase(), style: { flex: 1 } });
  const name = h('input.input#name', { value: s.name, placeholder: '예: 성현' });
  const theme = h('select.select#theme',
    ['system', 'light', 'dark'].map((v) => h('option', { value: v, selected: s.theme === v }, { system: '시스템 설정 따르기', light: '밝게', dark: '어둡게' }[v])));
  const status = h('span.chip');
  const paint = () => {
    const c = state.connection;
    status.textContent = c.server === 'ok' ? '연결됨' : '연결 안 됨';
    status.className = `chip ${c.server === 'ok' ? 'chip-ok' : 'chip-err'}`;
  };
  const off = on('connection', paint);
  paint();

  const toggle = (key, title, desc) => h('label.switch-row',
    h('div', h('div.t', title), h('div.d', desc)),
    h('input', { type: 'checkbox', checked: s[key], onchange: (e) => saveSettings({ [key]: e.target.checked }) }));

  root.append(h('div.container', { style: { maxWidth: '760px' } },
    h('div.section-head', h('div', h('h2', { style: { fontSize: 'var(--fs-xl)' } }, '설정'), h('p', '이 브라우저에만 저장돼요.'))),
    h('div.card.card-pad', { style: { display: 'flex', flexDirection: 'column', gap: '16px' } },
      h('div.section-head', h('h2', '학습 서버'), status),
      h('div.field', h('label', { for: 'serverUrl' }, '서버 주소'),
        h('div', { style: { display: 'flex', gap: '8px' } }, url,
          h('button.btn.btn-primary', { onclick: async () => {
            saveSettings({ serverUrl: url.value.trim() });
            session.reconnect();
            await loadCurriculum();
            toast('서버 주소를 저장했어요.');
          } }, '저장'))),
      h('p.muted', { style: { fontSize: '13px' } }, '비워 두면 이 페이지를 연 서버를 씁니다. 다른 컴퓨터의 서버를 쓰려면 예: http://192.168.0.10:8000')),
    h('div.card.card-pad',
      h('h2', { style: { fontSize: 'var(--fs-l)', marginBottom: '6px' } }, '학습'),
      h('div.field', { style: { padding: '8px 0 14px' } }, h('label', { for: 'name' }, '이름'), h('div', { style: { display: 'flex', gap: '8px' } }, name,
        h('button.btn', { onclick: () => { saveSettings({ name: name.value.trim() }); toast('저장했어요.'); } }, '저장'))),
      toggle('showLandmarks', '손 관절 표시', '카메라 화면 위에 인식된 손 관절을 그려요.'),
      toggle('autoSendPause', '멈추면 자동 전송 (프리토킹)', '수어를 마치고 잠시 멈추면 선생님께 전달해요.')),
    h('div.card.card-pad',
      h('h2', { style: { fontSize: 'var(--fs-l)', marginBottom: '6px' } }, '화면'),
      h('div.switch-row', h('div', h('div.t', '테마')), h('div', theme)))));

  theme.addEventListener('change', () => {
    saveSettings({ theme: theme.value });
    applyTheme(theme.value);
  });
  return off;
}

export function applyTheme(v) {
  if (v === 'system') document.documentElement.removeAttribute('data-theme');
  else document.documentElement.dataset.theme = v;
}
