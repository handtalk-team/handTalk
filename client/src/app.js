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

import { GloveMock } from './glove_mock.js';
import { CameraCapture } from './camera.js?v=2';
import { WSClient } from './websocket_client.js';
import { UI } from './ui.js';

const WS_URL = `ws://${location.host}/ws`;

// ─── instantiate ───────────────────────────────────────────────
const glove  = new GloveMock();
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
    fpsCounter = 0;
    fpsTime = now;
  }

  // Always update the confidence bar (even outside sessions)
  if (visionData) {
    ui.setVisionConfidence(visionData.confidence);
  } else {
    ui.setVisionConfidence(0);
  }

  if (!sessionActive || !ws.isOpen()) return;

  const gloveData = glove.read();
  ui.updateGloveDisplay(gloveData);

  const frame = {
    type: 'frame',
    data: {
      timestamp: Date.now() / 1000,
      sequence: frameSeq++,
      session_id: ws.sessionId ?? 'unknown',
      camera: visionData,
      glove: gloveData,
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

// ─── boot ──────────────────────────────────────────────────────
(async () => {
  glove.start();
  ws.connect();  // WebSocket은 카메라와 무관하게 먼저 연결
  try {
    await camera.start();
  } catch (err) {
    console.warn('카메라 초기화 실패:', err);
    ui.addSystemMsg('카메라를 시작할 수 없습니다. 카메라 권한을 확인하세요.', 'warning');
  }
})();
