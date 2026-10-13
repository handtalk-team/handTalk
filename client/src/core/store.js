/** 앱 전역 상태와 이벤트. 화면(view)들은 여기만 바라본다. */

const listeners = new Map();

export const state = {
  connection: { server: 'connecting', camera: 'off', glove: 'off', avatar: 'off' },
  curriculum: null,
  session: null,          // { mode, scenario, title, startedAt, lesson, utterance, transcript[] }
};

export function on(event, fn) {
  if (!listeners.has(event)) listeners.set(event, new Set());
  listeners.get(event).add(fn);
  return () => listeners.get(event).delete(fn);
}

export function emit(event, payload) {
  listeners.get(event)?.forEach((fn) => {
    try { fn(payload); } catch (e) { console.error(`[store] ${event}`, e); }
  });
}

export function setConnection(key, value) {
  if (state.connection[key] === value) return;
  state.connection[key] = value;
  emit('connection', state.connection);
}

// ── 설정 (브라우저에만 저장) ─────────────────────────────────────

const SETTINGS_KEY = 'handtalk.settings.v1';
const DEFAULTS = {
  serverUrl: '',          // 비우면 이 페이지를 연 서버 (FastAPI /app 으로 열었을 때)
  theme: 'system',
  showLandmarks: true,
  autoSendPause: true,
  speechCaptions: true,
  name: '',
  liveTutor: true,       // 실시간 메타휴먼(Pixel Streaming)이 있으면 사용
  avatarStreamUrl: '',   // 비우면 http://<이 컴퓨터>:8080
};

export function loadSettings() {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}') };
  } catch {
    return { ...DEFAULTS };
  }
}

export function saveSettings(patch) {
  const next = { ...loadSettings(), ...patch };
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(next)); } catch { /* 사생활 보호 모드 */ }
  emit('settings', next);
  return next;
}

export function serverBase() {
  const s = loadSettings().serverUrl.trim().replace(/\/$/, '');
  if (s) return s;
  if (location.protocol.startsWith('http') && location.port === '8000') return location.origin;
  return 'http://localhost:8000';
}

export function wsUrl(path = '/ws') {
  return serverBase().replace(/^http/, 'ws') + path;
}
