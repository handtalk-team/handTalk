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

const GLB_URL = 'assets/avatar/web/tutor.glb';        // 몸 + 옷 (원본 메시에서 내보냄, 스킨 가중치 정확)
const FACE_URL = 'assets/avatar/web/tutor_face.glb';  // 얼굴 (렌더 데이터에서 내보냄: 원본 메시 내보내기에는 얼굴이 빠진다)
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
  #ratio = 1; #maxRatio = 1; #frames = 0; #fpsT0 = 0;

  constructor(el) {
    this.#el = el;
    el.style.background = '#cfd2d3';
    const r = this.#renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance', preserveDrawingBuffer: !!window.__AVATAR_DEBUG });
    this.#maxRatio = Math.min(window.devicePixelRatio || 1, 1.5);
    this.#ratio = Math.min(this.#maxRatio, 1.25);   // 처음엔 조금 낮게 시작해서 프레임에 맞춰 조절
    r.setPixelRatio(this.#ratio);
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
    const load = (url) => new Promise((res, rej) => new THREE.GLTFLoader().load(url, res, undefined, rej));
    const [gltf, ref, face] = await Promise.all([load(GLB_URL), fetch(REF_URL).then((r) => r.json()), load(FACE_URL).catch(() => null)]);
    if (face) {
      // 얼굴 파일에서는 얼굴 메시만 쓴다. 머리·목은 움직이지 않으므로 얼굴 뼈대는 따로 움직일 필요가 없다.
      const drop = [];
      face.scene.traverse((o) => { if (o.isMesh && !/^Face/.test(o.name)) drop.push(o); else if (o.isMesh) o.frustumCulled = false; });
      drop.forEach((o) => o.parent.remove(o));
      // 눈물막·눈가 그림자·연골 같은 반투명 보조 메시는 웹 재질로 표현이 안 돼 검게 보이므로 숨긴다
      face.scene.traverse((o) => {
        if (o.isMesh && /Occlusion|lacrimal|Cartilage|EyeEdge|Eyelash/i.test(o.material?.name || '')) o.visible = false;
      });
      this.#scene.add(face.scene);
      await this.#addHair(face);
    }
    this.#ref = ref;
    const root = gltf.scene;
    this.#scene.add(root);
    root.updateMatrixWorld(true);

    // GLTFLoader 는 같은 이름의 노드에 _1, _2 를 붙이므로 (메시마다 뼈대가 따로 있음) 원래 본 이름을 되살린다
    const nodes = gltf.parser.json.nodes;
    root.traverse((o) => {
      const a = gltf.parser.associations.get(o);
      if (o.isBone && a && a.nodes !== undefined && nodes[a.nodes].name) o.userData.boneName = nodes[a.nodes].name;
    });
    const skins = [];
    root.traverse((o) => { if (o.isSkinnedMesh) { skins.push(o); o.frustumCulled = false; } });
    // 화면에는 머리부터 허리까지만 나오므로 신발은 그리지 않는다 (내장 그래픽 부담 줄이기)
    skins.forEach((s) => { if (/shs_|shoe|sneaker/i.test(s.name)) s.visible = false; });
    // 좌표 변환은 Body 처럼 뼈가 가장 많은 메시로 찾는다
    const main = skins.reduce((a, b) => (b.skeleton.bones.length > a.skeleton.bones.length ? b : a), skins[0]);
    const ue = [], gl = [];
    const v = new THREE.Vector3();
    for (const b of main.skeleton.bones) {
      const nm = b.userData.boneName || b.name;
      if (!ref.pos[nm]) continue;
      ue.push(ref.pos[nm]);
      b.getWorldPosition(v);
      gl.push([v.x, v.y, v.z]);
    }
    this.#axis = fitAxisMap(ue, gl);
    console.info('[web-avatar] axis fit rms', this.#axis.rms.toFixed(4), 'scale', this.#axis.scale.toFixed(4), 'skins', skins.length);

    const seen = new Set();   // 얼굴처럼 여러 메시가 한 뼈대를 같이 쓰면 한 번만
    for (const sk of skins) { if (!seen.has(sk.skeleton)) { seen.add(sk.skeleton); this.#rigs.push(this.#buildRig(sk)); } }
    this.#frame(root);
    this.#connect();
    this.#loop();
  }

  /** 머리카락·눈썹: 그룸 대신 카드 메시 + 투명도(coverage) 텍스처. 얼굴 컴포넌트 공간에 붙인다 (머리는 움직이지 않음). */
  async #addHair(face) {
    let anchor = face.scene;
    face.scene.traverse((o) => { if (o.name === 'Face' && !o.isMesh) anchor = o; });
    const loader = new THREE.GLTFLoader();
    const texLoader = new THREE.TextureLoader();
    const parts = [['hair', 0x1c120c, 'tex/hair_coverage.png'], ['eyebrows', 0x1a110b, 'tex/eyebrows_coverage.png']];
    await Promise.all(parts.map(async ([name, color, cov]) => {
      try {
        const [g, alpha] = await Promise.all([
          new Promise((res, rej) => loader.load(`assets/avatar/web/${name}.glb`, res, undefined, rej)),
          texLoader.loadAsync(`assets/avatar/web/${cov}`),
        ]);
        alpha.flipY = false;   // glTF UV 기준
        const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.85, metalness: 0, alphaMap: alpha, alphaTest: 0.35, side: THREE.DoubleSide });
        g.scene.traverse((o) => { if (o.isMesh) { o.material = mat; o.frustumCulled = false; } });
        anchor.add(g.scene);
      } catch (e) { console.warn(`[web-avatar] ${name} 없음`, e); }
    }));
  }

  /** Unreal 회전 → GLB 월드 회전 */
  #toWorld(ueQ, out) {
    this.#m.makeRotationFromQuaternion(ueQ);
    this.#m.premultiply(this.#axis.M).multiply(this.#axis.Mt);
    return out.setFromRotationMatrix(this.#m);
  }

  #buildRig(skin) {
    // 뼈대 맨 위 본("root")부터 모든 자식 노드를 부모 → 자식 순서로. 스킨의 joints 목록에 없는 중간 노드도
    // 부모 회전을 이어 주므로 같이 계산한다 (빠뜨리면 손가락이 원래 자리에 남아 피부가 늘어난다).
    const bones = new Set(skin.skeleton.bones);
    let top = skin.skeleton.bones[0];
    while (top.parent && (bones.has(top.parent) || top.parent.isBone)) top = top.parent;
    const order = [];
    const visit = (node, parentEntry) => {
      const name = node.userData.boneName || node.name;
      const refQ = this.#ref.rot[name];
      let offset = null;
      if (refQ) {
        // 같은 본이 참조 포즈에서 갖는 회전 차이 (GLB 월드 = 변환(UE 회전) × offset)
        const w = new THREE.Quaternion(); node.getWorldQuaternion(w);
        const u = this.#toWorld(new THREE.Quaternion(...refQ), new THREE.Quaternion());
        offset = u.invert().multiply(w);
      }
      const entry = { node, name, parent: parentEntry, offset, bindQ: node.quaternion.clone(), world: new THREE.Quaternion() };
      order.push(entry);
      node.children.forEach((c) => { if (!c.isMesh) visit(c, entry); });
    };
    visit(top, null);
    const rootParentQ = new THREE.Quaternion();
    if (top.parent) top.parent.getWorldQuaternion(rootParentQ);
    return { order, rootParentQ };
  }

  /** 머리부터 허리까지(수어하는 공간)가 화면에 차게 카메라를 맞춘다 */
  #frame(root) {
    let head = null, pelvis = null;
    root.traverse((o) => {
      const n = o.userData.boneName;
      if (n === 'head' && !head) head = o;
      if (n === 'pelvis' && !pelvis) pelvis = o;
    });
    const h = new THREE.Vector3(), p = new THREE.Vector3();
    if (head && pelvis) { head.getWorldPosition(h); pelvis.getWorldPosition(p); }
    else { const box = new THREE.Box3().setFromObject(root); h.set(0, box.max.y, 0); p.set(0, box.min.y + (box.max.y - box.min.y) * 0.53, 0); }
    const span = (h.y - p.y) * 1.35;                 // 머리 위 여유 ~ 허리
    const target = new THREE.Vector3(h.x, h.y + 0.22 - span / 2, h.z);
    const dist = (span / 2) / Math.tan(THREE.MathUtils.degToRad(this.#camera.fov / 2));
    this.#camera.position.set(target.x, target.y, target.z + dist);   // 메타휴먼 GLB 는 +Z 가 앞
    this.#camera.lookAt(target);
    this.#resize();
  }

  #connect() {
    if (this.#closed) return;
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = this.#ws = new WebSocket(`${proto}://${location.host}/ws/avatar`);
    ws.onmessage = (e) => {
      let msg; try { msg = JSON.parse(e.data); } catch { return; }
      if (msg.type === 'motion' && msg.bones?.length) this.playMotion(msg);
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
        const parentW = o.parent ? o.parent.world : rig.rootParentQ;
        const t = target.get(o.name);
        if (t && o.offset) {
          this.#toWorld(t, wq).multiply(o.offset);           // 원하는 월드 회전
          o.world.copy(wq);
          o.node.quaternion.copy(parentW).invert().multiply(wq);
        } else {
          o.node.quaternion.copy(o.bindQ);
          o.world.copy(parentW).multiply(o.bindQ);
        }
      }
    }
  }

  #loop() {
    if (this.#closed) return;
    this.#raf = requestAnimationFrame(() => this.#loop());
    const now = performance.now();
    this.#applyPose(now / 1000);
    this.#renderer.render(this.#scene, this.#camera);
    this.#adapt(now);
  }

  /** 내장 그래픽 대응: 2초마다 프레임을 재서 느리면 렌더 해상도를 낮추고, 여유가 있으면 다시 올린다 */
  #adapt(now) {
    if (!this.#fpsT0) { this.#fpsT0 = now; this.#frames = 0; return; }
    this.#frames++;
    if (now - this.#fpsT0 < 2000) return;
    const fps = this.#frames * 1000 / (now - this.#fpsT0);
    this.#fpsT0 = now; this.#frames = 0;
    if (document.hidden) return;
    let r = this.#ratio;
    if (fps < 40) r = Math.max(0.6, r - 0.15);
    else if (fps > 57) r = Math.min(this.#maxRatio, r + 0.1);
    if (Math.abs(r - this.#ratio) > 0.01) {
      this.#ratio = r;
      this.#renderer.setPixelRatio(r);
      this.#resize();
    }
    this.fps = Math.round(fps);
  }

  #resize() {
    const w = this.#el.clientWidth || 1, h = this.#el.clientHeight || 1;
    this.#renderer.setSize(w, h, false);
    this.#camera.aspect = w / h;
    this.#camera.updateProjectionMatrix();
  }

  /** 점검용: 내부 상태 */
  debug() { return { rigs: this.#rigs, axis: this.#axis, scene: this.#scene }; }

  /** 점검용: 지금 동작의 t 초 자세를 바로 그린다 (화면이 숨겨져 애니메이션 프레임이 멈출 때) */
  renderAt(t) {
    const p = this.#player;
    const now = (p.start || 0) + t;
    this.#applyPose(now);
    this.#renderer.render(this.#scene, this.#camera);
  }

  /** 서버 없이 동작 메시지를 직접 넣는다 (점검용) */
  playMotion(msg) {
    const m = parseMotion(msg);
    if (!m.n) return;
    while (this.#pose.length < m.bones.length) { this.#pose.push(new THREE.Quaternion()); this.#tmp.push(new THREE.Quaternion()); }
    this.#player.play(m, performance.now() / 1000);
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
