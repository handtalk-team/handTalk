import { h, icon } from '../core/dom.js';
import { state, loadSettings } from '../core/store.js';
import { summary, loadHistory } from '../core/history.js';
import { modeTitle } from '../core/api.js';

function fmtDate(iso) {
  const d = new Date(iso);
  return `${d.getMonth() + 1}월 ${d.getDate()}일 ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

export function render(root) {
  const s = summary();
  const last = loadHistory()[0];
  const name = loadSettings().name;
  const cur = state.curriculum;
  const firstDeck = cur?.decks?.[0]?.id || 'basics';

  const primary = last
    ? h('a.btn.btn-primary.btn-lg', { href: `#/session/${last.mode}/${last.scenario}` }, `이어서 하기 — ${last.title}`)
    : h('a.btn.btn-primary.btn-lg', { href: `#/session/word/${firstDeck}` }, '첫 수업 시작하기');

  root.append(h('div.container',
    h('section.hero',
      h('div.hero-text',
        h('div.kicker', name ? `${name}님, 다시 오셨네요` : 'AI 수어 선생님'),
        h('h2', '손으로 말하면,', h('br'), '선생님이 수어로 답해요'),
        h('p', '카메라 앞에서 수어를 하면 바로 알아보고, 메타휴먼 선생님이 수어와 자막으로 대답합니다. 단어부터 상황별 문장, 자유 대화까지 내 속도에 맞춰 연습하세요.'),
        h('div.actions', primary, h('a.btn.btn-lg', { href: '#/learn/free_talk' }, '선생님과 대화하기'))),
      h('figure.hero-media', { style: { margin: 0 } },
        h('img', { src: 'assets/brand/tutor.png', alt: '수어로 "감사합니다"를 표현하는 AI 선생님' }),
        h('figcaption', h('span', h('b', 'AI 선생님'), ' · 실시간 수어'), h('span.sign', '“감사합니다”')))),

    h('section.kpis',
      kpi(s.streak, '일', '연속 학습'),
      kpi(s.minutes, '분', '누적 학습 시간'),
      kpi(s.words, '개', '익힌 단어'),
      kpi(s.accuracy ?? '—', s.accuracy == null ? '' : '%', '평균 인식 정확도')),

    h('section',
      h('div.section-head', h('div', h('h2', '학습 과정'), h('p', '처음이라면 단어 연습부터, 익숙하다면 바로 대화로 넘어가세요.'))),
      h('div.mode-list', { style: { marginTop: '18px' } },
        (cur?.modes || []).map((m, i) => h('a.mode-item', { href: `#/learn/${m.id}` },
          h('span.mode-num', `${String(i + 1).padStart(2, '0')} · ${m.level}`),
          h('h3', m.title),
          h('p', m.summary),
          h('div.foot', h('span', courseCount(m.id)), h('span.go', '시작하기', icon('arrow', 15))))))),

    h('section',
      h('div.section-head', h('div', h('h2', '최근 학습')), h('a.btn.btn-ghost.btn-sm', { href: '#/history' }, '전체 기록', icon('arrow', 14))),
      h('div', { style: { marginTop: '10px', borderTop: '1px solid var(--line)' } }, recent()))));
}

function kpi(v, unit, label) {
  return h('div.kpi', h('div.v.num', String(v), unit ? h('small', unit) : null), h('div.l', label));
}

function courseCount(mode) {
  const c = state.curriculum;
  if (!c) return '';
  if (mode === 'word') return `단어 묶음 ${c.decks.length}개`;
  if (mode === 'sentence') return `상황 ${c.scenarios.length}개`;
  return `대화 주제 ${c.topics.length}개`;
}

function recent() {
  const list = loadHistory().slice(0, 5);
  if (!list.length) return h('div.empty', '아직 학습 기록이 없어요. 첫 수업을 시작해 보세요.');
  return h('div.list', list.map((r) => h('div.list-row',
    h('span.mode-tag', modeTitle(r.mode)),
    h('div.grow', h('div.t', r.title), h('div.d', fmtDate(r.date))),
    h('div.d.num', r.attempts ? `${r.correct} / ${r.attempts}` : ''),
    h('a.btn.btn-sm', { href: `#/session/${r.mode}/${r.scenario}` }, '다시 하기'))));
}
