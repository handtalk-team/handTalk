/**
 * 아바타 무대 — 튜터가 수어 클립을 재생하는 곳. 두 가지 방식 중 있는 것을 쓴다.
 *
 *   MetaHumanVideoStage  Unreal 메타휴먼으로 렌더한 클립 영상 (assets/avatar/{clip}.mp4, idle.jpg)
 *   AvatarStage          3D 미리보기 — 키포인트 모션 (assets/motions/{clip}.motion.json) 을 막대 인형으로 재생
 *
 *   LiveTutorStage       실시간 메타휴먼 — Unreal(TutorLive 레벨)이 서버 /ws/avatar 응답대로 움직이는 화면을
 *                        Pixel Streaming 으로 받아 보여 준다. 프리토킹처럼 정해지지 않은 대답도 바로 수어로 한다.
 *
 * 클립 이름은 서버 avatar_commands, Unreal /Game/HandTalk/Signs 와 같다.
 */
import { loadSettings } from '../core/store.js';

export function liveTutorUrl() {
  const s = loadSettings().avatarStreamUrl?.trim();
  return (s || `http://${location.hostname || 'localhost'}:8080`).replace(/\/$/, '');
}

/** 실시간 메타휴먼 → 메타휴먼 영상 → 3D 미리보기 순서로 쓸 수 있는 것을 고른다 */
export async function createAvatarStage(el, opts = {}) {
  if (loadSettings().liveTutor !== false) {
    try {
      const ctrl = new AbortController();
      setTimeout(() => ctrl.abort(), 1500);
      await fetch(`${liveTutorUrl()}/uiless.html`, { mode: 'no-cors', signal: ctrl.signal });
      return new LiveTutorStage(el, opts);
    } catch { /* 실시간 튜터 서버 없음 */ }
  }
  try {
    const res = await fetch('assets/avatar/idle.jpg', { method: 'HEAD' });
    if (res.ok) return new MetaHumanVideoStage(el, opts);
  } catch { /* 영상 없음 */ }
  return new AvatarStage(el, opts);
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

export class MetaHumanVideoStage {
  kind = 'metahuman';
  #el; #idle; #video; #queue = []; #onClip; #avail = new Map();
  #demo = []; #demoTimer = 0;

  constructor(el, { onClip } = {}) {
    this.#el = el;
    this.#onClip = onClip;
    const fit = { position: 'absolute', inset: '0', width: '100%', height: '100%', objectFit: 'cover', objectPosition: '50% 30%', transition: 'opacity .18s ease' };
    el.style.background = '#cfd2d3';   // 렌더 배경색과 맞춤
    this.#idle = Object.assign(document.createElement('img'), { src: 'assets/avatar/idle.jpg', alt: 'AI 선생님', className: 'avatar-idle' });
    this.#video = Object.assign(document.createElement('video'), { muted: true, playsInline: true, preload: 'auto' });
    Object.assign(this.#idle.style, fit);
    Object.assign(this.#video.style, fit, { opacity: '0' });
    this.#video.addEventListener('ended', () => this.#next());
    this.#video.addEventListener('playing', () => { this.#video.style.opacity = '1'; });
    el.prepend(this.#idle, this.#video);
  }

  async #has(clip) {
    if (!this.#avail.has(clip)) {
      let ok = false;
      try { ok = (await fetch(`assets/avatar/${clip}.mp4`, { method: 'HEAD' })).ok; } catch { /* 없음 */ }
      this.#avail.set(clip, ok);
    }
    return this.#avail.get(clip);
  }

  async #filter(clips) {
    const ok = [];
    for (const c of clips || []) if (c && await this.#has(c)) ok.push(c);
    return ok;
  }

  /** 한 번 재생하고, 끝나면 시범 반복(setDemo)으로 돌아간다 */
  async play(clips = []) {
    clearTimeout(this.#demoTimer);
    this.#queue = await this.#filter(clips);
    this.#next();
    return this.#queue.length;
  }

  /** 지금 연습할 단어/문장 시범을 잠깐씩 쉬며 계속 반복한다. 반환값: 재생할 수 있는 클립 수 */
  async setDemo(clips = []) {
    this.#demo = await this.#filter(clips);
    if (!this.#queue.length && this.#video.paused) this.#scheduleDemo(600);
    return this.#demo.length;
  }

  #scheduleDemo(ms = 1800) {
    clearTimeout(this.#demoTimer);
    if (!this.#demo.length) return;
    this.#demoTimer = setTimeout(() => {
      if (!this.#queue.length && this.#video.paused) { this.#queue = [...this.#demo]; this.#next(); }
    }, ms);
  }

  #next() {
    const clip = this.#queue.shift();
    this.#onClip?.(clip || null);
    if (!clip) {
      this.#video.style.opacity = '0';
      this.#video.pause();
      this.#scheduleDemo();
      return;
    }
    this.#video.src = `assets/avatar/${clip}.mp4`;
    this.#video.play().catch(() => this.#next());
  }

  destroy() { clearTimeout(this.#demoTimer); this.#video.pause(); this.#idle.remove(); this.#video.remove(); }
}
const BODY = [['r_shoulder', 'l_shoulder'], ['neck', 'mid_hip'], ['r_shoulder', 'r_elbow'], ['r_elbow', 'r_wrist'], ['l_shoulder', 'l_elbow'], ['l_elbow', 'l_wrist']];
const CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]];

// 클립이 없을 때 쓰는 기본 자세 (수어자 좌표: X 오른쪽, Y 앞, Z 위, 어깨 너비 1)
const REST = (() => {
  const j = { neck: [0, 0, 0], nose: [0, 0.35, 0.55], mid_hip: [0, 0, -1.6],
    r_shoulder: [0.5, 0, 0], l_shoulder: [-0.5, 0, 0],
    r_elbow: [0.62, 0.05, -0.75], l_elbow: [-0.62, 0.05, -0.75],
    r_wrist: [0.5, 0.35, -1.25], l_wrist: [-0.5, 0.35, -1.25] };
  for (const [h, w, s] of [['rh', j.r_wrist, 1], ['lh', j.l_wrist, -1]]) {
    j[`${h}_0`] = w;
    CHAINS.forEach((ch, f) => ch.slice(1).forEach((k, i) => {
      j[`${h}_${k}`] = [w[0] + s * (f - 2) * 0.035, w[1] + 0.02, w[2] - 0.06 - i * 0.045];
    }));
  }
  return j;
})();

export class AvatarStage {
  kind = '3d';
  #el; #renderer; #scene; #camera; #segs = []; #joints = []; #head;
  #cache = new Map(); #queue = []; #clip = null; #frame = 0; #acc = 0; #raf = 0; #last = 0;
  #pose = REST; #from = null; #blend = 1; #onClip = null;

  constructor(el, { onClip } = {}) {
    this.#el = el;
    this.#onClip = onClip;
    if (typeof THREE === 'undefined') {
      el.appendChild(Object.assign(document.createElement('div'), {
        className: 'empty', textContent: '3D 아바타를 불러오지 못했습니다. 인터넷 연결을 확인하세요.' }));
      return;
    }
    this.#init();
  }

  #init() {
    const r = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    r.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.#el.appendChild(r.domElement);
    this.#renderer = r;
    this.#scene = new THREE.Scene();
    this.#camera = new THREE.PerspectiveCamera(30, 1, 0.1, 50);
    this.#camera.position.set(0, -0.2, 8.6);
    this.#camera.lookAt(0, -0.55, 0);
    this.#scene.add(new THREE.HemisphereLight(0xffffff, 0xb9cfcc, 0.95));
    const key = new THREE.DirectionalLight(0xffffff, 0.55);
    key.position.set(2, 3, 5);
    this.#scene.add(key);

    const skin = new THREE.MeshStandardMaterial({ color: 0xf1c7a6, roughness: 0.65 });
    const shirt = new THREE.MeshStandardMaterial({ color: 0x13968a, roughness: 0.8 });
    const handR = new THREE.MeshStandardMaterial({ color: 0xf0bb95, roughness: 0.6 });
    const cyl = new THREE.CylinderGeometry(1, 1, 1, 14);
    const sph = new THREE.SphereGeometry(1, 20, 14);
    const add = (a, b, r, m) => {
      const mesh = new THREE.Mesh(cyl, m); this.#scene.add(mesh); this.#segs.push({ a, b, r, mesh });
      const jm = new THREE.Mesh(sph, m); jm.scale.setScalar(r * 1.05); this.#scene.add(jm); this.#joints.push([jm, b]);
    };
    // 몸통(셔츠)
    this.#torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.42, 0.9, 8, 20), shirt);
    this.#scene.add(this.#torso);
    BODY.forEach(([a, b]) => add(a, b, a.includes('shoulder') && b.includes('shoulder') ? 0.1 : 0.068, shirt));
    for (const h of ['rh', 'lh']) CHAINS.forEach((ch) => ch.slice(1).forEach((k, i) => add(`${h}_${ch[i]}`, `${h}_${k}`, 0.022, handR)));
    this.#head = new THREE.Mesh(sph, skin);
    this.#head.scale.set(0.2, 0.25, 0.21);
    this.#scene.add(this.#head);
    this.#neck = new THREE.Mesh(cyl, skin);
    this.#scene.add(this.#neck);

    new ResizeObserver(() => this.#resize()).observe(this.#el);
    this.#resize();
    this.#last = performance.now();
    const loop = (t) => { this.#tick(t); this.#raf = requestAnimationFrame(loop); };
    this.#raf = requestAnimationFrame(loop);
  }

  #torso; #neck;

  destroy() {
    cancelAnimationFrame(this.#raf);
    this.#renderer?.dispose();
    this.#el.replaceChildren();
  }

  /** 클립 이름 목록을 차례로 재생. 없는 클립은 건너뛴다. */
  async play(clips = []) {
    const loaded = [];
    for (const c of clips) {
      const m = await this.#load(c);
      if (m) loaded.push({ name: c, ...m });
    }
    this.#queue = loaded;
    this.#next();
    return loaded.length;
  }

  async has(clip) { return !!(await this.#load(clip)); }

  #demo = [];
  #demoWait = 0;

  /** 지금 연습할 단어/문장 시범을 계속 반복 */
  async setDemo(clips = []) {
    const loaded = [];
    for (const c of clips || []) {
      const m = await this.#load(c);
      if (m) loaded.push({ name: c, ...m });
    }
    this.#demo = loaded;
    return loaded.length;
  }

  async #load(clip) {
    if (!clip) return null;
    if (this.#cache.has(clip)) return this.#cache.get(clip);
    let m = null;
    try {
      const res = await fetch(`assets/motions/${clip}.motion.json`);
      if (res.ok) {
        const d = await res.json();
        m = { fps: d.fps || 30, joints: d.joints, frames: d.frames };
      }
    } catch { /* 클립 없음 */ }
    this.#cache.set(clip, m);
    return m;
  }

  #next() {
    if (!this.#queue.length && this.#demo.length && this.#clip) {
      this.#demoWait = 1.8;   // 한 번 끝나면 잠깐 쉬었다가 시범 반복
    }
    this.#from = this.#pose;
    this.#blend = 0;
    this.#clip = this.#queue.shift() || null;
    this.#frame = 0;
    this.#acc = 0;
    this.#onClip?.(this.#clip?.name || null);
  }

  #frameJoints(clip, f) {
    const row = clip.frames[Math.min(f, clip.frames.length - 1)];
    const j = {};
    clip.joints.forEach((n, i) => { j[n] = [row[3 * i], row[3 * i + 1], row[3 * i + 2]]; });
    return j;
  }

  #tick(now) {
    const dt = Math.min((now - this.#last) / 1000, 0.1);
    this.#last = now;
    let target;
    if (this.#clip) {
      this.#acc += dt * this.#clip.fps;
      while (this.#acc >= 1) { this.#acc -= 1; this.#frame += 1; }
      if (this.#frame >= this.#clip.frames.length + 8) this.#next();
      target = this.#clip ? this.#frameJoints(this.#clip, this.#frame) : REST;
    } else {
      if (this.#demo.length) {
        this.#demoWait -= dt;
        if (this.#demoWait <= 0) { this.#queue = [...this.#demo]; this.#next(); }
      }
      // 대기: 살짝 숨쉬기
      const b = Math.sin(now / 900) * 0.012;
      target = {};
      for (const k in REST) target[k] = [REST[k][0], REST[k][1], REST[k][2] + (k === 'mid_hip' ? 0 : b)];
    }
    this.#blend = Math.min(1, this.#blend + dt * 5);
    const pose = {};
    for (const k in target) {
      const a = this.#from?.[k];
      const t = target[k];
      pose[k] = a && this.#blend < 1 ? a.map((v, i) => v + (t[i] - v) * this.#blend) : t;
    }
    this.#pose = pose;
    this.#apply(pose);
    this.#renderer.render(this.#scene, this.#camera);
  }

  #apply(j) {
    const V = (p) => new THREE.Vector3(-p[0], p[2], p[1]);   // 마주 보는 시점
    const up = new THREE.Vector3(0, 1, 0);
    for (const s of this.#segs) {
      if (!j[s.a] || !j[s.b]) continue;
      const A = V(j[s.a]); const B = V(j[s.b]);
      const d = B.clone().sub(A); const len = d.length();
      s.mesh.position.copy(A).addScaledVector(d, 0.5);
      s.mesh.scale.set(s.r, Math.max(len, 1e-4), s.r);
      if (len > 1e-6) s.mesh.quaternion.setFromUnitVectors(up, d.normalize());
    }
    for (const [m, n] of this.#joints) if (j[n]) m.position.copy(V(j[n]));
    const neck = V(j.neck); const hip = V(j.mid_hip);
    this.#torso.position.copy(neck).lerp(hip, 0.48);
    this.#torso.scale.set(1, (neck.distanceTo(hip) * 0.86) / 1.74, 0.55);
    const headPos = neck.clone().add(new THREE.Vector3(0, 0.4, 0.02));
    this.#head.position.copy(headPos);
    this.#neck.position.copy(neck).add(new THREE.Vector3(0, 0.14, 0));
    this.#neck.scale.set(0.09, 0.28, 0.09);
  }

  #resize() {
    const w = this.#el.clientWidth || 1;
    const hgt = this.#el.clientHeight || 1;
    this.#renderer.setSize(w, hgt, false);
    this.#camera.aspect = w / hgt;
    this.#camera.updateProjectionMatrix();
  }
}
