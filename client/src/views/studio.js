/**
 * 실시간 학습 화면 — 세 모드가 같은 레이아웃을 쓰고 오른쪽 목표 카드와 버튼만 달라진다.
 *   왼쪽: AI 선생님(아바타 + 자막 + 글로스)   가운데: 나(카메라 + 실시간 인식)   오른쪽: 목표, 대화 기록, 조작
 */
import { h, icon, clear, toast } from '../core/dom.js';
import { state, on, loadSettings, saveSettings } from '../core/store.js';
import { courseTitle, modeTitle } from '../core/api.js';
import { CameraCapture } from '../core/camera.js';
import * as session from '../core/session.js';
import { createAvatarStage } from '../components/avatar-stage.js';

export function render(root, { mode = 'word', id = '' }) {
  const title = courseTitle(mode, id);
  const offs = [];
  let camera = null;
  let avatar = null;
  let camOff = null;

  root.classList.add('fill');
  showGate();

  return () => {
    offs.forEach((f) => f());
    if (state.session?.active) session.endSession();
    camera?.stop();
    avatar?.destroy();
  };

  // ── 시작 전 점검 ─────────────────────────────────────────────
  function showGate() {
    offs.splice(0).forEach((f) => f());
    camera?.stop();
    avatar?.destroy();
    avatar = null;
    clear(root);
    const items = {
      server: h('div.check', h('span.s', icon('server', 12)), '학습 서버 연결'),
      camera: h('div.check', h('span.s', icon('camera', 12)), '카메라 사용 권한 (시작할 때 요청해요)'),
      avatar: h('div.check', h('span.s', icon('check', 12)), 'AI 선생님 준비'),
    };
    const startBtn = h('button.btn.btn-primary.btn-lg', { onclick: start }, icon('play'), '수업 시작');
    const refresh = () => {
      const ok = session.isServerOnline();
      items.server.toggleAttribute('data-ok', ok);
      items.avatar.toggleAttribute('data-ok', typeof THREE !== 'undefined');
      startBtn.disabled = !ok;
      startBtn.lastChild.textContent = ok ? '수업 시작' : '서버 연결을 기다리는 중';
    };
    offs.push(on('connection', refresh));
    root.append(h('div.session-gate', h('div.card.gate-card',
      h('div', h('span.chip.chip-brand', modeTitle(mode)), ' '),
      h('h2', title),
      h('p.muted', GUIDE[mode]),
      h('div.checklist', Object.values(items)),
      h('div', { style: { display: 'flex', gap: '10px', flexWrap: 'wrap' } }, startBtn,
        h('a.btn.btn-lg', { href: `#/learn/${mode}` }, '다른 과정')),
      h('p.muted', { style: { fontSize: '13px' } }, '조작: ', h('kbd', 'H'), ' 힌트 · ', h('kbd', 'R'), ' 다시 보기 · ', h('kbd', 'N'), ' 건너뛰기',
        mode === 'free_talk' ? [' · ', h('kbd', 'Enter'), ' 보내기'] : null, ' · ', h('kbd', 'Esc'), ' 끝내기'))));
    refresh();
  }

  async function start() {
    if (!session.startSession({ mode, scenario: id, title })) {
      toast('서버에 연결되지 않았습니다.', 'error');
      return;
    }
    showStudio();
    try {
      await camera.start();
      state.connection.camera = 'ok';
    } catch (e) {
      camOff.hidden = false;
      camOff.textContent = `카메라를 켤 수 없어요: ${e.message || e.name}. 브라우저의 카메라 권한을 확인하세요.`;
    }
  }

  // ── 학습 화면 ───────────────────────────────────────────────
  function showStudio() {
    clear(root);
    const speech = h('div.speech', '선생님이 준비 중이에요…');
    const glossRow = h('div.gloss-row');
    const badge = h('span.chip', 'AI 선생님');
    const demoNote = h('span.chip', { hidden: true, style: { marginLeft: '6px' } }, '이 단어는 시범 영상을 준비 중이에요');
    const stage = h('div.avatar-stage', h('div.stage-badge', badge, demoNote), h('div.stage-caption', glossRow, speech));

    const video = h('video', { autoplay: true, muted: true, playsinline: true });
    const canvas = h('canvas');
    camOff = h('div.cam-off', { hidden: true });
    const handsChip = h('span.chip', '손 인식 대기');
    const fpsChip = h('span.chip.num', '0 fps');
    const live = h('div.live-word', { hidden: true });
    const confBar = h('i', { style: { width: '0%' } });
    const camWrap = h('div.camera-wrap', video, canvas, camOff, h('div.cam-hud', handsChip, fpsChip), live);

    // 거울 모드: 선생님을 좌우로 뒤집어 보여 준다. 기본은 실제로 마주 본 모습(선생님 오른손이 화면 왼쪽).
    // 거울 모드에서는 선생님 오른손이 화면 오른쪽에 와서, 거울 보듯 같은 쪽 손으로 따라 하기 쉽다.
    const applyMirror = (on) => {
      stage.classList.toggle('mirrored', on);
      mirrorBtn.toggleAttribute('data-on', on);
      mirrorBtn.textContent = on ? '거울 모드 켬' : '거울 모드';
    };
    const mirrorBtn = h('button.chip.chip-btn', { title: '선생님을 좌우로 뒤집어 거울처럼 보기', onclick: () => {
      const on = !loadSettings().mirrorTutor;
      saveSettings({ mirrorTutor: on });
      applyMirror(on);
    } }, '거울 모드');
    const target = h('div.target-card');
    const transcript = h('div.transcript');
    let thinking = null;   // 프리토킹: 선생님이 답을 만드는 동안 보이는 말풍선
    const buffer = h('div.buffer', '수어를 하면 여기에 단어가 모여요. 잠시 멈추면 선생님께 전달돼요.');
    const controls = h('div.controls',
      h('button.btn', { onclick: () => session.command('hint') }, icon('hint', 16), '힌트'),
      h('button.btn', { onclick: () => session.command('repeat') }, icon('repeat', 16), '다시 보기'),
      mode !== 'free_talk' ? h('button.btn', { onclick: () => session.command('skip') }, icon('skip', 16), '건너뛰기') : null);
    const compose = h('div.compose',
      mode === 'free_talk' ? [buffer, h('div.controls',
        h('button.btn.btn-primary', { onclick: () => session.command('submit') }, icon('send', 16), '보내기'),
        h('button.btn', { onclick: () => session.command('clear') }, icon('x', 16), '지우기'))] : null,
      controls,
      h('button.btn.btn-ghost.btn-danger', { onclick: finish }, icon('stop', 16), '수업 끝내기'));

    root.append(h('div.studio', { style: { position: 'relative' } },
      h('div.stage-col',
        h('div.panel-head', h('div.who', h('span.avatar-dot'), 'AI 선생님'), h('div.head-actions', mirrorBtn, h('span.chip.chip-brand', modeTitle(mode)))),
        stage),
      h('div.stage-col',
        h('div.panel-head', h('div.who', h('span.me-dot'), loadSettings().name || '나'), h('span.chip', title)),
        camWrap,
        h('div.signal', h('div.meter', h('span', '손 인식 신뢰도'), h('div.progress', confBar)))),
      h('aside.side', target, transcript, compose)));

    applyMirror(!!loadSettings().mirrorTutor);
    createAvatarStage(stage, {
      onClip: (clip) => glossRow.querySelectorAll('.gloss').forEach((g) => g.toggleAttribute('data-active', g.dataset.clip === clip && !!clip)),
    }).then((a) => {
      avatar = a;
      badge.textContent = { web: 'AI 선생님 · 메타휴먼', live: 'AI 선생님 · 실시간' }[a.kind] || 'AI 선생님 · 준비 중';
      if (pendingClips) { a.play(pendingClips); pendingClips = null; }
      if (pendingDemo) { applyDemo(pendingDemo); pendingDemo = null; }
    });
    let pendingClips = null;
    let pendingDemo = null;
    // 단어/문장 모드: 지금 연습할 동작을 선생님이 잠깐씩 쉬며 계속 보여 준다
    const applyDemo = async (clips) => {
      if (!avatar) { pendingDemo = clips; return; }
      const n = await avatar.setDemo(clips);
      demoNote.hidden = mode === 'free_talk' || n > 0 || !clips;
    };
    camera = new CameraCapture(video, canvas);
    session.attachCamera(camera);

    // ── 서버 이벤트 → 화면 ──
    offs.push(on('tutor', (msg) => {
      speech.textContent = msg.text;
      clear(glossRow).append(...(msg.gloss || []).map((g, i) => h('span.gloss', { dataset: { clip: msg.avatar_commands?.[i]?.clip || '' } }, g)));
      const clips = (msg.avatar_commands || []).map((c) => c.clip);
      if (avatar) avatar.play(clips); else pendingClips = clips;
    }));
    offs.push(on('transcript', (e) => {
      if (e.who === 'tutor') { thinking?.remove(); thinking = null; }
      const cls = e.who === 'me' ? 'me' : e.who === 'sys' ? 'sys' : `tutor${e.kind === 'feedback' || e.kind === 'hint' ? '.feedback' : e.kind === 'correct' ? '.correct' : ''}`;
      transcript.append(h(`div.bubble.${cls}`,
        e.who === 'tutor' ? h('span.k', KIND[e.kind] || '선생님') : null,
        e.who === 'me' ? [h('span.k', '내 수어'), `${e.text} `, h('small', { style: { opacity: 0.7 } }, `${Math.round((e.confidence || 0) * 100)}%`)] : e.text));
      transcript.scrollTop = transcript.scrollHeight;
    }));
    offs.push(on('recognition', (msg) => {
      const expected = state.session?.lesson?.target;
      const nextGloss = expected ? expected.gloss[expected.matched || 0] : null;
      const good = mode === 'free_talk' || !nextGloss || msg.text === nextGloss;
      live.hidden = false;
      live.dataset.state = good ? 'ok' : 'bad';
      live.replaceChildren(msg.text, h('small.num', `${Math.round(msg.confidence * 100)}%`));
      clearTimeout(live._t);
      live._t = setTimeout(() => { live.hidden = true; }, 1800);
    }));
    offs.push(on('lesson', (l) => {
      renderTarget(target, l, mode);
      if (mode !== 'free_talk') applyDemo(l.done ? [] : (l.target?.clips || []));
    }));
    offs.push(on('utterance', (u) => {
      clear(buffer);
      if (!u.gloss.length) buffer.append('수어를 하면 여기에 단어가 모여요. 잠시 멈추면 선생님께 전달돼요.');
      else buffer.append(...u.gloss.map((g) => h('span.gloss', g)), u.final ? h('span.muted', ' 전달됨') : null);
      if (u.thinking) {
        thinking?.remove();
        thinking = h('div.bubble.tutor.thinking', h('span.k', '선생님'), h('span.dots', h('i'), h('i'), h('i')));
        transcript.append(thinking);
        transcript.scrollTop = transcript.scrollHeight;
      }
    }));
    offs.push(on('hands', ({ right, left }) => {
      const n = (right ? 1 : 0) + (left ? 1 : 0);
      handsChip.textContent = n ? `손 ${n}개 인식 중` : '손 인식 대기';
      const c = Math.max(right?.confidence ?? 0, left?.confidence ?? 0);
      confBar.style.width = `${Math.round(c * 100)}%`;
    }));
    offs.push(on('fps', (f) => { fpsChip.textContent = `${f} fps`; }));
    offs.push(on('session', (s) => { if (!s.active && s === state.session && !root.querySelector('.result-overlay')) showResult(); }));

    const keys = (e) => {
      if (e.target instanceof Element && e.target.matches('input, textarea, select')) return;
      const k = e.key.toLowerCase();
      if (k === 'h') session.command('hint');
      else if (k === 'r') session.command('repeat');
      else if (k === 'n' && mode !== 'free_talk') session.command('skip');
      else if (e.key === 'Enter' && mode === 'free_talk') session.command('submit');
      else if (e.key === 'Escape') finish();
    };
    document.addEventListener('keydown', keys);
    offs.push(() => document.removeEventListener('keydown', keys));
  }

  function finish() {
    session.endSession();
    camera?.stop();
  }

  function showResult() {
    const s = state.session;
    const l = s?.lesson;
    const correct = l?.correct ?? 0;
    const attempts = l?.attempts ?? 0;
    const secs = Math.round((Date.now() - (s?.startedAt || Date.now())) / 1000);
    const pct = attempts ? Math.round((correct / attempts) * 100) : 0;
    root.querySelector('.studio')?.append(h('div.result-overlay', h('div.card.result-card',
      h('div.eyebrow', '수업 결과'),
      h('h2', title),
      h('div.big.num', mode === 'free_talk' ? `${correct}턴` : `${pct}%`),
      h('p.muted', mode === 'free_talk' ? `선생님과 ${correct}번 주고받았어요 · ${Math.max(1, Math.round(secs / 60))}분`
        : `${attempts}번 시도 중 ${correct}번 정답 · ${Math.max(1, Math.round(secs / 60))}분`),
      s?.learned?.size ? h('div.gloss-row', [...s.learned].map((g) => h('span.gloss', { dataset: { done: '' } }, g))) : null,
      h('div', { style: { display: 'flex', gap: '8px', justifyContent: 'center', flexWrap: 'wrap' } },
        h('button.btn.btn-primary', { onclick: () => { avatar?.destroy(); avatar = null; showGate(); } }, '다시 하기'),
        h('a.btn', { href: `#/learn/${mode}` }, '다른 과정'),
        h('a.btn.btn-ghost', { href: '#/' }, '홈으로')))));
  }
}

const GUIDE = {
  word: '선생님의 시범을 보고 같은 동작을 카메라 앞에서 해 보세요. 손 전체가 화면에 들어오게 앉아 주세요.',
  sentence: '상대방의 말을 읽고, 대답할 문장을 수어 순서대로 표현하세요. 단어 하나를 마칠 때마다 잠깐 멈추면 정확도가 올라가요.',
  free_talk: '자유롭게 수어로 말하세요. 말을 마치고 2~3초 멈추면 선생님께 전달되고, 선생님이 수어와 자막으로 답해요.',
};

const KIND = { prompt: '선생님', correct: '정답', feedback: '교정', hint: '힌트', reply: '선생님' };

function renderTarget(el, l, mode) {
  clear(el);
  const pct = l.total ? Math.round(((l.step - (l.done ? 0 : 1)) / l.total) * 100) : 0;
  if (mode !== 'free_talk') {
    el.append(h('div.step-row', h('span.num', `${l.step} / ${l.total}`), h('div.progress', h('i', { style: { width: `${l.done ? 100 : pct}%` } })),
      h('span.num', l.attempts ? `정답 ${l.correct}` : '')));
  } else {
    el.append(h('div.step-row', h('span.chip', l.title), h('span.num', l.step ? `${l.step}번째 대화` : '대화 시작')));
  }
  if (l.partner_line) el.append(h('div.partner', h('b', mode === 'free_talk' ? '선생님: ' : '상대방: '), l.partner_line));
  const t = l.target;
  if (t) {
    if (mode === 'word') {
      el.append(h('div.eyebrow', '따라 할 단어'), h('div.word', t.text));
    } else {
      el.append(h('div.eyebrow', '표현할 문장'), h('div', { style: { fontSize: '18px', fontWeight: 700 } }, t.text),
        h('div.gloss-row', { style: { justifyContent: 'flex-start' } }, t.gloss.map((g, i) => h('span.gloss',
          { dataset: i < t.matched ? { done: '' } : i === t.matched ? { active: '' } : {} }, g))));
    }
    if (t.description && mode === 'word') el.append(h('div.desc', t.description));
  }
  if (l.done) el.append(h('span.chip.chip-ok', '이 과정을 모두 마쳤어요'));
}
