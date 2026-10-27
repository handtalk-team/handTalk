/**
 * handTalk 웹 앱 — 진입점
 *
 *   #/                       홈 (요약, 모드, 최근 학습)
 *   #/learn/:mode            과정 선택 (word | sentence | free_talk)
 *   #/session/:mode/:id      실시간 학습
 *   #/history                학습 기록
 *   #/devices                기기 점검 (카메라, 장갑 색 보정, 센서 장갑)
 *   #/settings               설정
 *   #/lab                    개발자 도구 (데이터 수집)
 */
import { h, icon, clear } from './core/dom.js';
import { state, on, loadSettings } from './core/store.js';
import { loadCurriculum } from './core/api.js';
import * as session from './core/session.js';
import { applyTheme } from './views/settings.js';

const VIEWS = {
  home: () => import('./views/home.js'),
  learn: () => import('./views/learn.js'),
  session: () => import('./views/studio.js'),
  history: () => import('./views/history.js'),
  settings: () => import('./views/settings.js'),
  devices: () => import('./views/devices.js'),
  lab: () => import('./views/lab.js'),
};

const TITLES = { home: '홈', learn: '학습', session: '수업', history: '학습 기록', settings: '설정', devices: '기기 점검', lab: '개발자 도구' };
const MODE_NAMES = { word: '단어 연습', sentence: '상황별 문장', free_talk: '프리토킹' };

let cleanup = null;
let shell, viewEl, titleEl, crumbEl;

function parse() {
  const parts = location.hash.replace(/^#\/?/, '').split('/').filter(Boolean).map(decodeURIComponent);
  const name = parts[0] || 'home';
  if (name === 'learn') return { name, params: { mode: parts[1] || 'word' } };
  if (name === 'session') return { name, params: { mode: parts[1] || 'word', id: parts[2] || '' } };
  return { name: VIEWS[name] ? name : 'home', params: {} };
}

async function route() {
  const { name, params } = parse();
  if (typeof cleanup === 'function') cleanup();
  cleanup = null;
  shell.removeAttribute('data-nav-open');
  clear(viewEl);
  viewEl.className = 'view';
  titleEl.textContent = name === 'learn' ? MODE_NAMES[params.mode] : TITLES[name];
  crumbEl.textContent = name === 'session' ? MODE_NAMES[params.mode] : '';
  document.querySelectorAll('.nav-link').forEach((a) => {
    const target = a.getAttribute('href');
    const current = name === 'learn' || name === 'session' ? `#/learn/${params.mode}` : `#/${name === 'home' ? '' : name}`;
    if (target === current) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
  });
  const mod = await VIEWS[name]();
  cleanup = mod.render(viewEl, params) || null;
  viewEl.focus({ preventScroll: true });
}

function pill(key, label, ico) {
  const el = h('span.pill', { dataset: { state: 'off' }, title: label }, h('span.dot'), icon(ico, 14), label);
  const paint = () => {
    const v = state.connection[key];
    el.dataset.state = v === 'ok' ? 'ok' : v === 'connecting' ? 'warn' : 'off';
  };
  on('connection', paint);
  paint();
  return el;
}

function buildShell() {
  const nav = (href, ico, label, tag) => h('a.nav-link', { href }, icon(ico), label, tag ? h('span.tag', tag) : null);
  titleEl = h('h1');
  crumbEl = h('span.crumb');
  viewEl = h('main.view', { tabindex: '-1' });
  shell = h('div.shell',
    h('nav.sidebar', { 'aria-label': '주 메뉴' },
      h('a.brand', { href: '#/' },
        h('span.brand-mark', icon('word', 20)),
        h('span', h('div.brand-name', 'handTalk'), h('div.brand-sub', 'AI 수어 튜터'))),
      nav('#/', 'home', '홈'),
      h('div.nav-group', h('span.eyebrow', '학습'),
        nav('#/learn/word', 'word', '단어 연습'),
        nav('#/learn/sentence', 'sentence', '상황별 문장'),
        nav('#/learn/free_talk', 'chat', '프리토킹')),
      h('div.nav-group', h('span.eyebrow', '내 학습'),
        nav('#/history', 'chart', '학습 기록'),
        nav('#/devices', 'camera', '기기 점검'),
        nav('#/settings', 'settings', '설정')),
      h('div.nav-group', h('span.eyebrow', '개발'),
        nav('#/lab', 'lab', '데이터 수집')),
      h('div.sidebar-foot', '카메라 영상은 서버로 보내지 않고, 손 관절 좌표만 전송해요.')),
    h('div.main',
      h('header.topbar',
        h('button.icon-btn.menu-btn', { 'aria-label': '메뉴', onclick: () => shell.toggleAttribute('data-nav-open') }, icon('menu')),
        titleEl, crumbEl, h('span.spacer'),
        h('div.status-pills', pill('server', '서버', 'server'), pill('glove', '장갑', 'glove'))),
      viewEl));
  document.body.replaceChildren(shell);
}

applyTheme(loadSettings().theme);
buildShell();
session.connect();
on('connection', (c) => { if (c.server === 'ok' && state.curriculum?.offline) loadCurriculum().then(route); });
loadCurriculum().then(() => {
  window.addEventListener('hashchange', route);
  route();
});
