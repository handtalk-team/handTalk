/**
 * 웹 메타휴먼 — 언리얼 없이 브라우저(three.js)에서 메타휴먼이 수어한다.
 *
 * 서버가 문장마다 동작 생성 모델로 만든 동작({"type":"motion"}, /ws/avatar)을 받아, GLB 로 내보낸
 * 메타휴먼(assets/avatar/web/tutor.glb)의 뼈대에 그대로 입힌다. 클립은 쓰지 않는다.
 *
 * 서버가 보내는 회전은 Unreal 컴포넌트 공간 기준(Body 메시 참조 포즈, assets/avatar/ref_pose.json)이다.
 * GLB 는 좌표계가 달라서, 불러올 때 두 참조 포즈의 본 위치를 맞춰 좌표 변환(축 바꾸기)을 찾고,
 * 본마다 기준 회전 차이(offset)를 구해 둔다. 몸과 옷 메시는 각자 같은 동작을 받는다 (Unreal 과 같은 방식).
 */

const GLB_URL = 'assets/avatar/web/tutor.glb';
const REF_URL = 'assets/avatar/ref_pose.json';
const BLEND_S = 0.25;

/** GLB 가 있으면 웹 메타휴먼을 쓸 수 있다 */
export async function webAvatarAvailable() {
  if (typeof THREE === 'undefined' || !THREE.GLTFLoader) return false;
  try { return (await fetch(GLB_URL, { method: 'HEAD' })).ok; } catch { return false; }
}

// ── 동작 재생 (Unreal FHandTalkLivePlayer 와 같은 규칙) ──────────────────

function sampleMotion(m, elapsed, out) {
  const nb = m.bones.length;
  let t = elapsed;
  if (m.loop && m.n > 1) t %= m.duration + m.loopGap;
  const f = Math.min(Math.max(t * m.fps, 0), m.n - 1);
  const a = Math.floor(f);
  const b = Math.min(a + 1, m.n - 1);
  const w = f - a;
  for (let i = 0; i < nb; i++) {
    const qa = m.q[a * nb + i];
    out[i].copy(qa);
    if (w > 0) out[i].slerp(m.q[b * nb + i], w);
  }
}

class MotionPlayer {
  current = null; previous = null; pending = null;
  start = 0; prevStart = 0;

  #finished(m, now, start) { return !m.loop && now - start >= m.duration; }

  play(m, now) {
    const busy = this.current && !this.current.loop && !this.#finished(this.current, now, this.start);
    if (m.loop && busy) { this.pending = m; return; }   // 시범·쉬는 자세는 지금 수어가 끝난 뒤
    if (!m.loop) this.pending = null;
    this.previous = this.current; this.prevStart = this.start;
    this.current = m; this.start = now;
  }

  /** 지금 자세 (Unreal 회전, 본 순서는 current.bones) */
  sample(now, out, tmp) {
    if (this.pending && (!this.current || this.#finished(this.current, now, this.start))) {
      this.previous = this.current; this.prevStart = this.start;
      this.current = this.pending; this.start = now; this.pending = null;
    }
    if (!this.current) return null;
    sampleMotion(this.current, now - this.start, out);
    const blend = (now - this.start) / BLEND_S;
    if (blend < 1 && this.previous && this.previous.bones.length === this.current.bones.length) {
      sampleMotion(this.previous, now - this.prevStart, tmp);
      const w = blend * blend * (3 - 2 * blend);
      for (let i = 0; i < out.length; i++) out[i].copy(tmp[i].slerp(out[i], w));
    } else {
      this.previous = null;
    }
    return this.current.bones;
  }
}

function parseMotion(msg) {
  const nb = msg.bones.length;
  const q = [];
  for (const fr of msg.frames) {
    if (fr.length !== nb * 4) continue;
    for (let i = 0; i < nb; i++) q.push(new THREE.Quaternion(fr[i * 4], fr[i * 4 + 1], fr[i * 4 + 2], fr[i * 4 + 3]));
  }
  const n = q.length / nb;
  const fps = msg.fps || 30;
  return { bones: msg.bones, q, n, fps, loop: !!msg.loop, loopGap: msg.loop_gap || 0, duration: n > 1 ? (n - 1) / fps : 0 };
}

// ── 좌표 변환: Unreal 컴포넌트 공간 → GLB 월드 공간 ────────────────────────

/** 부호 있는 축 바꾸기 48가지 중 두 참조 포즈의 본 위치가 가장 잘 맞는 것 (크기·이동 포함) */
function fitAxisMap(uePts, glPts) {
  const perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
  const mean = (pts) => pts.reduce((s, p) => s.map((v, i) => v + p[i] / pts.length), [0, 0, 0]);
  const mu = mean(uePts), mg = mean(glPts);
  const U = uePts.map((p) => p.map((v, i) => v - mu[i]));
  const G = glPts.map((p) => p.map((v, i) => v - mg[i]));
  let best = null;
  for (const p of perms) {
    for (let s = 0; s < 8; s++) {
      const sg = [s & 1 ? -1 : 1, s & 2 ? -1 : 1, s & 4 ? -1 : 1];
      const map = (v) => [sg[0] * v[p[0]], sg[1] * v[p[1]], sg[2] * v[p[2]]];
      let num = 0, den = 0;
      U.forEach((u, k) => { const m = map(u); num += m[0] * G[k][0] + m[1] * G[k][1] + m[2] * G[k][2]; den += m[0] ** 2 + m[1] ** 2 + m[2] ** 2; });
      const scale = num / den;
      if (scale <= 0) continue;
      let err = 0;
      U.forEach((u, k) => { const m = map(u); err += (scale * m[0] - G[k][0]) ** 2 + (scale * m[1] - G[k][1]) ** 2 + (scale * m[2] - G[k][2]) ** 2; });
      if (!best || err < best.err) best = { err, p, sg, scale };
    }
  }
  // 행렬 M: gl = scale * M * ue  (M 은 부호 있는 순열 → 직교)
  const M = new THREE.Matrix4();
  const e = new Array(16).fill(0);
  for (let r = 0; r < 3; r++) e[best.p[r] * 4 + r] = best.sg[r];   // column-major: (row r, col p[r])
  e[15] = 1;
  M.fromArray(e);
  return { M, Mt: M.clone().transpose(), rms: Math.sqrt(best.err / U.length), scale: best.scale };
}

export class WebAvatarStage {
  kind = 'web';
  #el; #renderer; #scene; #camera; #raf = 0; #ws = null; #closed = false;
  #player = new MotionPlayer();
  #rigs = [];        // 메시마다 { order: [{bone, parent, idx, offset}], rootParentQ }
  #ref = null; #axis = null;
  #pose = []; #tmp = [];
  #m = new THREE.Matrix4(); #q = new THREE.Quaternion();

  constructor(el) {
    this.#el = el;
    el.style.background = '#cfd2d3';
    const r = this.#renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));   // 내장 그래픽 고려
    r.outputEncoding = THREE.sRGBEncoding;
    Object.assign(r.domElement.style, { position: 'absolute', inset: '0', width: '100%', height: '100%' });
    el.prepend(r.domElement);
    this.#scene = new THREE.Scene();
    this.#scene.background = new THREE.Color(0xcfd2d3);
    this.#scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8f94, 0.9));
    const key = new THREE.DirectionalLight(0xffffff, 1.1);
    key.position.set(0.6, 1.6, 2.2);
    this.#scene.add(key);
    this.#camera = new THREE.PerspectiveCamera(28, 1, 0.05, 50);
    this.ready = this.#load();
    new ResizeObserver(() => this.#resize()).observe(el);
  }

  async #load() {
    const [gltf, ref] = await Promise.all([
      new Promise((res, rej) => new THREE.GLTFLoader().load(GLB_URL, res, undefined, rej)),
      fetch(REF_URL).then((r) => r.json()),
    ]);
    this.#ref = ref;
    const root = gltf.scene;
    this.#scene.add(root);
    root.updateMatrixWorld(true);

    const skins = [];
    root.traverse((o) => { if (o.isSkinnedMesh) { skins.push(o); o.frustumCulled = false; } });
    // 좌표 변환은 Body 처럼 뼈가 가장 많은 메시로 찾는다
    const main = skins.reduce((a, b) => (b.skeleton.bones.length > a.skeleton.bones.length ? b : a), skins[0]);
    const ue = [], gl = [];
    const v = new THREE.Vector3();
    for (const b of main.skeleton.bones) {
      if (!ref.pos[b.name]) continue;
      ue.push(ref.pos[b.name]);
      b.getWorldPosition(v);
      gl.push([v.x, v.y, v.z]);
    }
    this.#axis = fitAxisMap(ue, gl);
    console.info('[web-avatar] axis fit rms', this.#axis.rms.toFixed(4), 'scale', this.#axis.scale.toFixed(4), 'skins', skins.length);

    for (const sk of skins) this.#rigs.push(this.#buildRig(sk));
    this.#frame(root);
    this.#connect();
    this.#loop();
  }

  /** Unreal 회전 → GLB 월드 회전 */
  #toWorld(ueQ, out) {
    this.#m.makeRotationFromQuaternion(ueQ);
    this.#m.premultiply(this.#axis.M).multiply(this.#axis.Mt);
    return out.setFromRotationMatrix(this.#m);
  }

  #buildRig(skin) {
    const bones = skin.skeleton.bones;
    const set = new Set(bones);
    const order = [];
    const visit = (b) => {
      const parent = set.has(b.parent) ? b.parent : null;
      const refQ = this.#ref.rot[b.name];
      let offset = null;
      if (refQ) {
        // 같은 본이 참조 포즈에서 갖는 회전 차이 (GLB 월드 = 변환(UE 회전) × offset)
        const w = new THREE.Quaternion(); b.getWorldQuaternion(w);
        const u = this.#toWorld(new THREE.Quaternion(...refQ), new THREE.Quaternion());
        offset = u.invert().multiply(w);
      }
      order.push({ bone: b, parent, offset, bindQ: b.quaternion.clone(), world: new THREE.Quaternion() });
      b.children.forEach((c) => { if (set.has(c)) visit(c); });
    };
    bones.filter((b) => !set.has(b.parent)).forEach(visit);
    const rootParentQ = new THREE.Quaternion();
    if (order[0]?.bone.parent) order[0].bone.parent.getWorldQuaternion(rootParentQ);
    const byName = new Map(order.map((o) => [o.bone.name, o]));
    return { order, byName, rootParentQ };
  }

  /** 상반신이 화면에 차게 카메라를 맞춘다 */
  #frame(root) {
    const box = new THREE.Box3().setFromObject(root);
    const size = box.getSize(new THREE.Vector3());
    const top = box.max.y;
    const target = new THREE.Vector3((box.min.x + box.max.x) / 2, top - size.y * 0.26, (box.min.z + box.max.z) / 2);
    const front = new THREE.Vector3(0, 0, 1);
    // 얼굴이 향한 쪽: 코 본이 있으면 그 방향, 없으면 +Z
    this.#camera.position.copy(target).addScaledVector(front, size.y * 1.05);
    this.#camera.lookAt(target);
    this.#resize();
  }

  #connect() {
    if (this.#closed) return;
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = this.#ws = new WebSocket(`${proto}://${location.host}/ws/avatar`);
    ws.onmessage = (e) => {
      let msg; try { msg = JSON.parse(e.data); } catch { return; }
      if (msg.type === 'motion' && msg.bones?.length) {
        const m = parseMotion(msg);
        if (!m.n) return;
        while (this.#pose.length < m.bones.length) { this.#pose.push(new THREE.Quaternion()); this.#tmp.push(new THREE.Quaternion()); }
        this.#player.play(m, performance.now() / 1000);
      }
    };
    ws.onclose = () => { if (!this.#closed) setTimeout(() => this.#connect(), 2000); };
  }

  #applyPose(now) {
    const bones = this.#player.sample(now, this.#pose, this.#tmp);
    if (!bones) return;
    const target = new Map();
    bones.forEach((name, i) => target.set(name, this.#pose[i]));
    const wq = new THREE.Quaternion();
    for (const rig of this.#rigs) {
      for (const o of rig.order) {
        const parentW = o.parent ? rig.byName.get(o.parent.name).world : rig.rootParentQ;
        const t = target.get(o.bone.name);
        if (t && o.offset) {
          this.#toWorld(t, wq).multiply(o.offset);           // 원하는 월드 회전
          o.world.copy(wq);
          o.bone.quaternion.copy(parentW).invert().multiply(wq);
        } else {
          o.bone.quaternion.copy(o.bindQ);
          o.world.copy(parentW).multiply(o.bindQ);
        }
      }
    }
  }

  #loop() {
    if (this.#closed) return;
    this.#raf = requestAnimationFrame(() => this.#loop());
    this.#applyPose(performance.now() / 1000);
    this.#renderer.render(this.#scene, this.#camera);
  }

  #resize() {
    const w = this.#el.clientWidth || 1, h = this.#el.clientHeight || 1;
    this.#renderer.setSize(w, h, false);
    this.#camera.aspect = w / h;
    this.#camera.updateProjectionMatrix();
  }

  // 동작은 서버가 /ws/avatar 로 직접 보낸다 (llm_response, lesson 마다 motion)
  async play() { return 1; }
  async setDemo(clips = []) { return clips ? 1 : 0; }

  destroy() {
    this.#closed = true;
    cancelAnimationFrame(this.#raf);
    this.#ws?.close();
    this.#renderer.dispose();
    this.#renderer.domElement.remove();
  }
}
