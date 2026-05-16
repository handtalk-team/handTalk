/**
 * UI — all DOM manipulation lives here.
 * Keeps app.js clean and focused on data flow.
 */

export class UI {
  #chatBox      = document.getElementById('chatBox');
  #feedbackLog  = document.getElementById('feedbackLog');
  #statTotal    = document.getElementById('statTotal');
  #statAcc      = document.getElementById('statAcc');
  #statLatency  = document.getElementById('statLatency');
  #statFPS      = document.getElementById('statFPS');
  #latencyChip  = document.getElementById('latencyChip');
  #wsBadge      = document.getElementById('wsBadge');
  #totalSigns   = 0;
  #correctSigns = 0;
  #latencies    = [];

  setWsBadge(connected) {
    this.#wsBadge.textContent = connected ? '서버: 연결됨' : '서버: 끊김';
    this.#wsBadge.className   = connected ? 'badge badge-live' : 'badge badge-dis';
  }

  setSessionButtons(active) {
    document.getElementById('btnStart').disabled = active;
    document.getElementById('btnEnd').disabled   = !active;
  }

  setFPS(fps) { this.#statFPS.textContent = fps; }

  setLatency(ms) {
    this.#latencies.push(ms);
    if (this.#latencies.length > 30) this.#latencies.shift();
    const avg = this.#latencies.reduce((a, b) => a + b, 0) / this.#latencies.length;
    this.#statLatency.textContent = Math.round(avg);
    this.#latencyChip.textContent = `⏱ ${Math.round(avg)} ms`;
  }

  setVisionConfidence(conf) {
    const pct = Math.round(conf * 100);
    document.getElementById('confVisionBar').style.width = `${pct}%`;
    document.getElementById('confVisionVal').textContent = `${pct}%`;
  }

  updateGloveDisplay(g) {
    if (!g) return;
    g.flex.forEach((v, i) => {
      document.getElementById(`f${i}`).style.height = `${Math.round(v * 100)}%`;
    });
    const fmt = (v) => (v >= 0 ? '+' : '') + v.toFixed(2);
    document.getElementById('ax').textContent = fmt(g.imu.accel[0]);
    document.getElementById('ay').textContent = fmt(g.imu.accel[1]);
    document.getElementById('az').textContent = fmt(g.imu.accel[2]);
    document.getElementById('gx').textContent = g.imu.gyro[0].toFixed(3);
    document.getElementById('gy').textContent = g.imu.gyro[1].toFixed(3);
    document.getElementById('gz').textContent = g.imu.gyro[2].toFixed(3);
    const q = Math.round(g.ble_quality * 100);
    document.getElementById('confGloveBar').style.width = `${q}%`;
    document.getElementById('confGloveVal').textContent = `${q}%`;
  }

  addSystemMsg(text, level = 'info') {
    const el = document.createElement('div');
    el.className = 'msg msg-sys';
    el.textContent = text;
    if (level === 'error')   el.style.color = '#f87171';
    if (level === 'warning') el.style.color = '#f59e0b';
    this.#chatBox.appendChild(el);
    this.#chatBox.scrollTop = this.#chatBox.scrollHeight;
  }

  onRecognition(msg) {
    this.#totalSigns++;
    this.#statTotal.textContent = this.#totalSigns;

    const pct = Math.round(msg.confidence * 100);
    const el = document.createElement('div');
    el.className = msg.is_partial ? 'msg msg-user msg-partial' : 'msg msg-user';
    el.textContent = `${msg.text}  (${pct}%)`;

    if (!msg.is_partial && msg.confidence >= 0.7) {
      this.#correctSigns++;
    }
    this.#statAcc.textContent =
      `${Math.round((this.#correctSigns / this.#totalSigns) * 100)}%`;

    this.#chatBox.appendChild(el);
    this.#chatBox.scrollTop = this.#chatBox.scrollHeight;
  }

  onLLMResponse(msg) {
    const el = document.createElement('div');
    el.className = 'msg msg-ai';
    el.textContent = msg.text;
    this.#chatBox.appendChild(el);
    this.#chatBox.scrollTop = this.#chatBox.scrollHeight;

    // Log avatar commands to console for UE5 debugging
    if (msg.avatar_commands?.length) {
      console.log('[Avatar]', msg.avatar_commands);
    }
  }

  onFeedback(msg) {
    if (!msg.errors?.length && !msg.suggestions?.length) return;
    const card = document.createElement('div');
    card.className = 'fb-card';

    const errors = (msg.errors || [])
      .map(e => `<div class="fb-error">⚠ [${e.part}] ${e.description}</div>`)
      .join('');
    const tips = (msg.suggestions || [])
      .map(s => `<div class="fb-tip">✓ ${s}</div>`)
      .join('');
    const dtw = msg.dtw_score != null
      ? `<div class="fb-dtw">DTW: ${msg.dtw_score.toFixed(2)}</div>`
      : '';

    card.innerHTML = `${errors}${tips}${dtw}`;
    this.#feedbackLog.prepend(card);

    // Keep last 20 feedback cards
    while (this.#feedbackLog.children.length > 20) {
      this.#feedbackLog.lastChild.remove();
    }
  }

  onSummary(msg) {
    const el = document.createElement('div');
    el.className = 'msg msg-sys';
    el.innerHTML = `
      📊 세션 종료 — 전체 수어: ${msg.total_signs}
      | 정확도: ${Math.round(msg.accuracy * 100)}%
    `;
    this.#chatBox.appendChild(el);
    this.#chatBox.scrollTop = this.#chatBox.scrollHeight;
  }
}
