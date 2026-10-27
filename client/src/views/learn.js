import { h, icon } from '../core/dom.js';
import { state } from '../core/store.js';

const INTRO = {
  word: ['단어 연습', '선생님이 먼저 동작을 보여 주면 따라 하세요. 맞히면 다음 단어로 넘어가고, 틀리면 손 모양과 움직임을 짚어 드려요.'],
  sentence: ['상황별 문장', '상대방의 말에 맞는 문장을 수어 어순대로 표현해요. 실제 장면을 따라가며 짧은 대화를 완성합니다.'],
  free_talk: ['프리토킹', '정해진 답 없이 선생님과 대화해요. 수어를 하고 잠시 멈추면 말이 전달되고, 선생님이 수어와 자막으로 답합니다.'],
};

export function render(root, { mode = 'word' }) {
  const c = state.curriculum;
  const [title, desc] = INTRO[mode] || INTRO.word;
  root.append(h('div.container', { style: { gap: '36px' } },
    h('div.page-head', h('h2', title), h('p', desc)),
    c?.offline ? h('div.card.card-pad', { style: { background: 'var(--warn-bg)', borderColor: 'transparent' } },
      '서버에 연결되지 않아 기본 과정만 보여요. 설정에서 서버 주소를 확인하세요.') : null,
    h('div.course-grid', mode === 'word' ? c.decks.map(deckRow)
      : mode === 'sentence' ? c.scenarios.map(scenarioRow)
        : c.topics.map(topicRow))));
}

function startBtn(mode, id) {
  return h('a.btn', { href: `#/session/${mode}/${id}` }, '시작', icon('arrow', 15));
}

function deckRow(d) {
  const ok = d.words.filter((w) => w.available).length;
  return h('div.course',
    h('div.title-col', h('h3', d.title), h('span.meta', `단어 ${ok}개${ok < d.words.length ? ` · ${d.words.length - ok}개 준비 중` : ''}`)),
    h('div.body.words', d.words.map((w) => h('span', { dataset: w.available ? {} : { off: '' }, title: w.available ? '' : '인식 모델에 아직 없는 단어' }, w.gloss))),
    h('div.cta', startBtn('word', d.id)));
}

function scenarioRow(s) {
  return h('div.course',
    h('div.title-col', h('h3', s.title), h('span.meta', `${s.place} · 대화 ${s.turns.length}번`)),
    h('div.body', h('div', s.goal), h('div.turns', s.turns.slice(0, 2).map((t) => h('div', t.partner)))),
    h('div.cta', startBtn('sentence', s.id)));
}

function topicRow(t) {
  return h('div.course',
    h('div.title-col', h('h3', t.title), h('span.meta', '자유 대화')),
    h('div.body', `“${t.opener}”`),
    h('div.cta', startBtn('free_talk', t.id)));
}
