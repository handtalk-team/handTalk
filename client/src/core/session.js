/**
 * 학습 세션 컨트롤러 — 서버 연결, 카메라 프레임 전송, 서버 메시지를 상태/이벤트로 바꾼다.
 *
 * 서버 → 클라이언트 메시지 (server/app/models/schemas/ws_messages.py)
 *   lesson        현재 단계 (목표 단어/문장, 진행률, 상대 대사)
 *   llm_response  튜터의 말 + 수어 글로스 + 아바타 클립
 *   recognition   사용자의 수어 인식 결과
 *   utterance     (프리토킹) 보내기 전까지 모은 글로스
 *   feedback      손 모양/움직임 오류
 *   session_summary, glove_status, system
 */
import { WSClient } from './ws.js';
import { state, emit, setConnection, wsUrl, loadSettings } from './store.js';
import { addRecord } from './history.js';

const ws = new WSClient(wsUrl());
let camera = null;
let seq = 0;
let fpsCount = 0;
let fpsT = performance.now();

export function connect() {
  ws.on('open', () => setConnection('server', 'ok'));
  ws.on('close', () => {
    setConnection('server', 'off');
    if (state.session?.active) {
      state.session.active = false;
      emit('session', state.session);
    }
  });
  ws.on('message', onMessage);
  ws.connect();
}

export function reconnect() {
  ws.setUrl(wsUrl());
}

export function isServerOnline() { return ws.isOpen(); }

/** 카메라를 세션 프레임 송신원으로 연결 */
export function attachCamera(cam) {
  camera = cam;
  cam.drawLandmarks = loadSettings().showLandmarks;
  cam.onFrame(({ right, left }) => {
    fpsCount += 1;
    const now = performance.now();
    if (now - fpsT >= 1000) {
      emit('fps', fpsCount);
      fpsCount = 0;
      fpsT = now;
    }
    emit('hands', { right, left });
    if (!state.session?.active || !ws.isOpen()) return;
    ws.send({
      type: 'frame',
      data: {
        timestamp: Date.now() / 1000,
        sequence: seq++,
        session_id: state.session.serverId || 'unknown',
        camera: right ?? null,
        camera_left: left ?? null,
        glove: null,
      },
    });
  });
}

export function startSession({ mode, scenario, words, title }) {
  if (!ws.isOpen()) return false;
  seq = 0;
  state.session = {
    mode, scenario, title, active: true, startedAt: Date.now(),
    lesson: null, utterance: [], transcript: [], correct: 0, attempts: 0, learned: new Set(),
    serverId: null,
  };
  ws.send({ type: 'start_session', mode, scenario, words: words || null });
  emit('session', state.session);
  return true;
}

export function command(action) {
  ws.send({ type: 'tutor_command', action });
}

export function endSession({ save = true } = {}) {
  const s = state.session;
  if (!s) return null;
  if (s.active) ws.send({ type: 'end_session' });
  s.active = false;
  const seconds = Math.round((Date.now() - s.startedAt) / 1000);
  const result = {
    date: new Date().toISOString(),
    mode: s.mode, scenario: s.scenario, title: s.title,
    correct: s.lesson?.correct ?? s.correct, attempts: s.lesson?.attempts ?? s.attempts,
    seconds, learned: [...s.learned],
  };
  if (save && seconds > 5) addRecord(result);
  emit('session', s);
  emit('session-ended', result);
  return result;
}

export function collectSample(label) {
  ws.send({ type: 'recording_mode', active: true });
  setTimeout(() => ws.send({ type: 'capture_sample', label }), 2000);
}

function push(entry) {
  const s = state.session;
  if (!s) return;
  s.transcript.push({ t: Date.now(), ...entry });
  emit('transcript', entry);
}

function onMessage(msg) {
  const s = state.session;
  switch (msg.type) {
    case 'lesson':
      if (s) {
        // 맞힌 단어 기록 (단어 모드: 다음 단계로 넘어가면 이전 목표를 익힌 것으로 본다)
        if (s.lesson?.target && msg.correct > (s.lesson.correct ?? 0)) s.lesson.target.gloss.forEach((g) => s.learned.add(g));
        s.lesson = msg;
      }
      emit('lesson', msg);
      break;
    case 'llm_response':
      push({ who: 'tutor', text: msg.text, kind: msg.kind, gloss: msg.gloss || [] });
      emit('tutor', msg);
      break;
    case 'recognition':
      if (s) s.attempts += 1;
      push({ who: 'me', text: msg.text, confidence: msg.confidence });
      emit('recognition', msg);
      break;
    case 'utterance':
      if (s) s.utterance = msg.final ? [] : msg.gloss;
      emit('utterance', msg);
      break;
    case 'feedback':
      emit('feedback', msg);
      break;
    case 'glove_status':
      setConnection('glove', msg.right || msg.left ? 'ok' : 'off');
      emit('glove', msg);
      break;
    case 'collect_ack':
      emit('collect', msg);
      break;
    case 'session_summary':
      emit('summary', msg);
      break;
    case 'system': {
      const m = /ID:\s*(\S+)/.exec(msg.message || '');
      if (m && s) s.serverId = m[1];
      if (msg.level !== 'info') push({ who: 'sys', text: msg.message, level: msg.level });
      emit('system', msg);
      break;
    }
    default:
      break;
  }
}
