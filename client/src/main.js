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
  clear(viewEl);
  viewEl.className = 'view';
  viewEl.scrollTop = 0;
  titleEl.textContent = name === 'learn' ? MODE_NAMES[params.mode] : TITLES[name];
  crumbEl.textContent = name === 'session' ? MODE_NAMES[params.mode] : '';
  document.querySelectorAll('.nav-link').forEach((a) => {
    const target = a.getAttribute('href');
    const current = name === 'learn' || name === 'session' ? `#/learn/${params.mode}` : `#/${name === 'home' ? '' : name}`;
    if (target === current) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
  });
  requestAnimationFrame(() => moveIndicator(shell.navEl));
  shell.querySelector('.topbar')?.removeAttribute('data-scrolled');
  const mod = await VIEWS[name]();
  cleanup = mod.render(viewEl, params) || null;
  viewEl.focus({ preventScroll: true });
}

function pill(key, label) {
  const el = h('span.pill', { dataset: { state: 'off' }, title: label }, h('span.dot'), h('span.pill-label', label));
  const paint = () => {
    const v = state.connection[key];
    el.dataset.state = v === 'ok' ? 'ok' : v === 'connecting' ? 'warn' : 'off';
    el.title = `${label} ${v === 'ok' ? '연결됨' : v === 'connecting' ? '연결 중' : '꺼짐'}`;
  };
  on('connection', paint);
  paint();
  return el;
}

/* 로고: 마주 보는 두 손(말을 주고받는 모양)을 두 개의 호로 그린 기호 + 소문자 글자 */
function logo() {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 32 32');
  svg.setAttribute('class', 'logo-mark');
  svg.setAttribute('aria-hidden', 'true');
  svg.innerHTML = '<path d="M13 6.5a9.5 9.5 0 0 0 0 19" /><path d="M19 25.5a9.5 9.5 0 0 0 0-19" /><circle cx="16" cy="16" r="2.6" />';
  return h('a.logo', { href: '#/', 'aria-label': 'handTalk 홈' }, svg, h('span.logo-word', 'hand', h('b', 'talk')));
}

/* 지금 페이지 밑줄이 링크 사이를 미끄러져 옮겨 간다 */
function moveIndicator(navEl) {
  const ind = navEl.querySelector('.nav-ind');
  const cur = navEl.querySelector('.nav-link[aria-current="page"]');
  if (!ind) return;
  if (!cur) { ind.style.opacity = '0'; return; }
  ind.style.opacity = '1';
  ind.style.width = `${cur.offsetWidth}px`;
  ind.style.transform = `translateX(${cur.offsetLeft}px)`;
}

function buildShell() {
  const nav = (href, label) => h('a.nav-link', { href }, label);
  const more = h('details.more',
    h('summary.icon-btn', { 'aria-label': '더 보기' }, icon('menu', 18)),
    h('div.more-menu',
      h('a.more-link', { href: '#/devices' }, icon('camera', 16), '기기 점검'),
      h('a.more-link', { href: '#/settings' }, icon('settings', 16), '설정'),
      h('a.more-link', { href: '#/lab' }, icon('lab', 16), '데이터 수집'),
      h('p.more-foot', '카메라 영상은 서버로 보내지 않고 손 관절 좌표만 보내요.')));
  more.addEventListener('click', (e) => { if (e.target.closest('a')) more.open = false; });
  document.addEventListener('click', (e) => { if (!more.contains(e.target)) more.open = false; });
  const navEl = h('nav.nav', { 'aria-label': '주 메뉴' },
    nav('#/', '홈'),
    nav('#/learn/word', '단어 연습'),
    nav('#/learn/sentence', '상황별 문장'),
    nav('#/learn/free_talk', '프리토킹'),
    nav('#/history', '학습 기록'),
    h('span.nav-ind', { 'aria-hidden': 'true' }));
  titleEl = h('h1.sr-only');
  crumbEl = h('span.sr-only');
  viewEl = h('main.view', { tabindex: '-1' });
  const bar = h('header.topbar',
    logo(),
    navEl,
    h('span.spacer'),
    titleEl, crumbEl,
    h('div.status-pills', pill('server', '서버'), pill('glove', '장갑')),
    more);
  viewEl.addEventListener('scroll', () => bar.toggleAttribute('data-scrolled', viewEl.scrollTop > 4), { passive: true });
  shell = h('div.shell', h('div.main', bar, viewEl));
  document.body.replaceChildren(shell);
  window.addEventListener('resize', () => moveIndicator(navEl));
  shell.navEl = navEl;
}

applyTheme(loadSettings().theme);
buildShell();
session.connect();
on('connection', (c) => { if (c.server === 'ok' && state.curriculum?.offline) loadCurriculum().then(route); });
loadCurriculum().then(() => {
  window.addEventListener('hashchange', route);
  route();
});
