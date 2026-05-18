/**
 * handTalk Web Client — Main Entry Point
 *
 * Wires together:
 *   GloveMock    → simulates ESP32 BLE sensor data (replaced by BLEGlove later)
 *   CameraCapture → webcam + MediaPipe Hands landmark extraction
 *   WSClient     → WebSocket connection to the FastAPI server
 *   UI           → DOM updates
 *
 * Data flow (30 Hz):
 *   CameraCapture.onFrame(landmarks)
 *     + GloveMock.read()
 *     → WSClient.sendFrame({ camera, glove, timestamp, sequence })
 *     ← server sends { type: "recognition" | "llm_response" | "feedback" | "system" }
 */

import { CameraCapture } from './camera.js?v=6';
import { WSClient } from './websocket_client.js';
import { UI } from './ui.js';

const WS_URL = `ws://${location.host}/ws`;

// ─── instantiate ───────────────────────────────────────────────
// glove = null  →  vision-only mode
// glove = new BLEGlove()  →  enable when ESP32 hardware is ready (see glove_mock.js)
const glove  = null;
const camera = new CameraCapture(
  document.getElementById('videoEl'),
  document.getElementById('canvasEl'),
);
const ws  = new WSClient(WS_URL);
const ui  = new UI();

let sessionActive = false;
let frameSeq = 0;
let fpsCounter = 0;
let fpsTime = performance.now();

// ─── WebSocket event routing ────────────────────────────────────
ws.on('open', () => {
  ui.setWsBadge(true);
  ui.addSystemMsg('서버 연결됨. 세션을 시작하세요.');
});

ws.on('close', () => {
  ui.setWsBadge(false);
  sessionActive = false;
  ui.setSessionButtons(false);
});

ws.on('message', (msg) => {
  switch (msg.type) {
    case 'recognition':
      ui.onRecognition(msg);
      break;
    case 'llm_response':
      ui.onLLMResponse(msg);
      break;
    case 'feedback':
      ui.onFeedback(msg);
      break;
    case 'session_summary':
      ui.onSummary(msg);
      sessionActive = false;
      ui.setSessionButtons(false);
      break;
    case 'collect_ack':
      onCollectAck(msg);
      break;
    case 'system':
      ui.addSystemMsg(msg.message, msg.level);
      break;
    default:
      console.warn('Unknown message type:', msg.type);
  }
});

// ─── Camera frame callback (~30 Hz) ────────────────────────────
camera.onFrame(async (visionData) => {
  // FPS tracking
  fpsCounter++;
  const now = performance.now();
  if (now - fpsTime >= 1000) {
    ui.setFPS(fpsCounter);
    if (fpsCounter === 0) console.warn('[handTalk] camera callback not firing!');
    fpsCounter = 0;
    fpsTime = now;
  }

  // Always update the confidence bar (even outside sessions)
  if (visionData) {
    ui.setVisionConfidence(visionData.confidence);
    if (sessionActive) ui.setHandDetecting(true);
  } else {
    ui.setVisionConfidence(0);
    if (sessionActive) ui.setHandDetecting(false);
  }

  if (!sessionActive || !ws.isOpen()) {
    if (sessionActive) console.warn('[handTalk] frame skipped — ws not open', ws.isOpen());
    return;
  }

  const frame = {
    type: 'frame',
    data: {
      timestamp: Date.now() / 1000,
      sequence: frameSeq++,
      session_id: ws.sessionId ?? 'unknown',
      camera: visionData,
      glove: glove ? glove.read() : null,
    },
  };

  const t0 = performance.now();
  ws.send(frame);
  ui.setLatency(performance.now() - t0);
});

// ─── UI button handlers ─────────────────────────────────────────
document.getElementById('btnStart').addEventListener('click', () => {
  if (!ws.isOpen()) {
    ui.addSystemMsg('서버에 연결되지 않았습니다.', 'error');
    return;
  }
  const scenario = document.getElementById('scenarioSel').value;
  ws.send({ type: 'start_session', scenario });
  sessionActive = true;
  frameSeq = 0;
  ui.setSessionButtons(true);
  ui.addSystemMsg(`세션 시작 — 시나리오: ${scenario}`);
});

document.getElementById('btnEnd').addEventListener('click', () => {
  ws.send({ type: 'end_session' });
  sessionActive = false;
  ui.setSessionButtons(false);
});

// ─── Data collection ────────────────────────────────────────────
const btnCapture = document.getElementById('btnCapture');
const collectLabelInput = document.getElementById('collectLabelInput');
const collectCounts = document.getElementById('collectCounts');
const recordBar = document.getElementById('recordBar');
const recordFill = document.getElementById('recordFill');
const labelCounts = {};
let recording = false;

function startRecording() {
  if (!ws.isOpen()) { ui.addSystemMsg('서버에 연결되지 않았습니다.', 'error'); return; }
  if (!sessionActive) { ui.addSystemMsg('세션을 먼저 시작하세요.', 'warning'); return; }
  const label = collectLabelInput.value.trim();
  if (!label) { ui.addSystemMsg('레이블을 입력하세요.', 'warning'); return; }
  if (recording) return;

  recording = true;
  ws.send({ type: 'recording_mode', active: true });
  btnCapture.textContent = '녹화 중...';
  btnCapture.style.background = '#ef4444';

  // 진행바 표시
  recordBar.style.display = 'block';
  recordFill.style.transition = 'none';
  recordFill.style.width = '0%';
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      recordFill.style.transition = 'width 2s linear';
      recordFill.style.width = '100%';
    });
  });

  setTimeout(() => {
    ws.send({ type: 'capture_sample', label });
    recording = false;
    btnCapture.textContent = '녹화 시작 [Space]';
    btnCapture.style.background = '#0f766e';
    recordBar.style.display = 'none';
    recordFill.style.width = '0%';
  }, 2000);
}

btnCapture.addEventListener('click', startRecording);

document.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && document.activeElement !== collectLabelInput) {
    e.preventDefault();
    startRecording();
  }
});

function onCollectAck(msg) {
  labelCounts[msg.label] = msg.count;
  collectCounts.innerHTML = Object.entries(labelCounts)
    .map(([l, c]) => `<span style="display:flex;justify-content:space-between"><span>${l}</span><span style="color:#a5b4fc;font-weight:600">${c}개</span></span>`)
    .join('');
}

// ─── boot ──────────────────────────────────────────────────────
(async () => {
  if (glove) await glove.start();
  ws.connect();  // WebSocket은 카메라와 무관하게 먼저 연결
  try {
    await camera.start();
  } catch (err) {
    console.warn('카메라 초기화 실패:', err);
    ui.addSystemMsg('카메라를 시작할 수 없습니다. 카메라 권한을 확인하세요.', 'warning');
  }
})();
