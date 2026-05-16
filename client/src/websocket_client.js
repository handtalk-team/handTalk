/**
 * WSClient — WebSocket wrapper with auto-reconnect.
 *
 * Emits events: 'open', 'close', 'message'
 */

export class WSClient {
  #url;
  #ws = null;
  #handlers = { open: [], close: [], message: [] };
  #reconnectDelay = 2000;
  sessionId = null;

  constructor(url) { this.#url = url; }

  connect() {
    this.#ws = new WebSocket(this.#url);
    this.#ws.onopen    = () => this.#emit('open');
    this.#ws.onclose   = () => {
      this.#emit('close');
      setTimeout(() => this.connect(), this.#reconnectDelay);
    };
    this.#ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        // Capture session ID from system messages
        if (msg.type === 'system' && msg.message?.includes('ID:')) {
          const m = msg.message.match(/ID:\s*(\S+)/);
          if (m) this.sessionId = m[1];
        }
        this.#emit('message', msg);
      } catch { /* ignore malformed */ }
    };
    this.#ws.onerror   = (e) => console.error('WS error', e);
  }

  send(data) {
    if (this.isOpen()) {
      this.#ws.send(JSON.stringify(data));
    }
  }

  isOpen() { return this.#ws?.readyState === WebSocket.OPEN; }

  on(event, fn) { this.#handlers[event]?.push(fn); }

  #emit(event, ...args) {
    this.#handlers[event]?.forEach(fn => fn(...args));
  }
}
