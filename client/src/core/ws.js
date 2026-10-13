/** 자동 재연결 WebSocket. 이벤트: open, close, message */
export class WSClient {
  #url;
  #ws = null;
  #handlers = { open: [], close: [], message: [] };
  #retry = 1000;
  #closedByUser = false;

  constructor(url) { this.#url = url; }

  setUrl(url) {
    if (url === this.#url) return;
    this.#url = url;
    this.#ws?.close();
  }

  connect() {
    this.#closedByUser = false;
    try {
      this.#ws = new WebSocket(this.#url);
    } catch (e) {
      console.warn('[ws] 연결 실패', e);
      this.#schedule();
      return;
    }
    this.#ws.onopen = () => { this.#retry = 1000; this.#emit('open'); };
    this.#ws.onclose = () => { this.#emit('close'); if (!this.#closedByUser) this.#schedule(); };
    this.#ws.onerror = () => {};
    this.#ws.onmessage = (e) => {
      try { this.#emit('message', JSON.parse(e.data)); } catch { /* 깨진 메시지 무시 */ }
    };
  }

  close() { this.#closedByUser = true; this.#ws?.close(); }

  send(data) { if (this.isOpen()) this.#ws.send(JSON.stringify(data)); }

  isOpen() { return this.#ws?.readyState === WebSocket.OPEN; }

  on(event, fn) { this.#handlers[event].push(fn); return () => { this.#handlers[event] = this.#handlers[event].filter((f) => f !== fn); }; }

  #schedule() {
    setTimeout(() => this.connect(), this.#retry);
    this.#retry = Math.min(this.#retry * 1.6, 8000);
  }

  #emit(event, ...args) { this.#handlers[event].forEach((fn) => fn(...args)); }
}
