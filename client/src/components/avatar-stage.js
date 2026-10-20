/**
 * 아바타 무대 — AI 선생님이 수어하는 곳.
 *
 *   WebAvatarStage   웹 메타휴먼 (web-avatar.js): 브라우저 three.js 가 서버의 수어 동작(/ws/avatar motion)을 메타휴먼에 입힌다
 *   LiveTutorStage   실시간 메타휴먼 (Unreal TutorLive + Pixel Streaming): 언리얼을 켜 둔 개발용
 *
 * 녹화 클립은 쓰지 않는다. 모든 동작은 서버의 동작 생성 모델이 문장마다 새로 만든다.
 */
import { loadSettings } from '../core/store.js';
import { WebAvatarStage, webAvatarAvailable } from './web-avatar.js';

export function liveTutorUrl() {
  const s = loadSettings().avatarStreamUrl?.trim();
  return (s || `http://${location.hostname || 'localhost'}:8080`).replace(/\/$/, '');
}

/**
 * 웹 메타휴먼(브라우저 three.js, 언리얼 없음) → 실시간 메타휴먼(Unreal Pixel Streaming) 순서로 고른다.
 * 녹화 클립을 재생하는 예전 방식(영상, 막대 인형)은 쓰지 않는다: 모든 동작은 서버의 동작 생성 모델이 만든다.
 */
export async function createAvatarStage(el, opts = {}) {
  if (await webAvatarAvailable()) {
    const stage = new WebAvatarStage(el, opts);
    try { await stage.ready; return stage; } catch (e) { console.warn('[web-avatar] 불러오기 실패', e); stage.destroy(); }
  }
  if (loadSettings().liveTutor !== false) {
    try {
      const ctrl = new AbortController();
      setTimeout(() => ctrl.abort(), 1500);
      await fetch(`${liveTutorUrl()}/uiless.html`, { mode: 'no-cors', signal: ctrl.signal });
      return new LiveTutorStage(el, opts);
    } catch { /* 실시간 튜터 서버 없음 */ }
  }
  return new PlaceholderStage(el, opts);
}

export class LiveTutorStage {
  kind = 'live';
  #frame;

  constructor(el) {
    el.style.background = '#cfd2d3';
    const q = new URLSearchParams({ AutoConnect: 'true', AutoPlayVideo: 'true', StartVideoMuted: 'true', HoveringMouse: 'true', MatchViewportRes: 'false' });
    this.#frame = Object.assign(document.createElement('iframe'), {
      src: `${liveTutorUrl()}/uiless.html?${q}`, title: '실시간 AI 선생님', allow: 'autoplay; fullscreen',
    });
    Object.assign(this.#frame.style, { position: 'absolute', inset: '0', width: '100%', height: '100%', border: '0' });
    el.prepend(this.#frame);
  }

  // 동작은 Unreal 이 서버에서 직접 받는다 (llm_response, lesson)
  async play() { return 1; }

  // 클립이 있는 단어만 시범을 보일 수 있다 (서버가 준 clips 가 비어 있으면 준비 중)
  async setDemo(clips = []) { return clips.length; }

  destroy() { this.#frame.remove(); }
}

/** 웹 메타휴먼 파일도, 실시간 튜터도 없을 때: 안내만 보여 준다 (녹화 클립은 재생하지 않음) */
export class PlaceholderStage {
  kind = 'none';
  #note;

  constructor(el) {
    el.style.background = '#cfd2d3';
    this.#note = Object.assign(document.createElement('div'), {
      textContent: 'AI 선생님 준비 중이에요. assets/avatar/web 의 메타휴먼 파일을 확인해 주세요.',
    });
    Object.assign(this.#note.style, { position: 'absolute', inset: '0', display: 'grid', placeItems: 'center', padding: '24px', textAlign: 'center', color: '#55606a', fontSize: '14px' });
    el.prepend(this.#note);
  }

  async play() { return 0; }
  async setDemo() { return 0; }
  destroy() { this.#note.remove(); }
}
