import { h, icon } from '../core/dom.js';
import { state } from '../core/store.js';

const INTRO = {
  word: ['단어 연습', '아바타 선생님이 먼저 동작을 보여 주면 따라 하세요. 맞히면 다음 단어로, 틀리면 손 모양과 움직임을 알려 드려요.'],
  sentence: ['상황별 문장', '상대방의 말에 맞는 문장을 수어 어순(글로스)대로 표현해요. 단어를 순서대로 맞히면 문장이 완성됩니다.'],
  free_talk: ['프리토킹', '정해진 답 없이 선생님과 대화해요. 수어를 하고 잠시 멈추면 말이 전달되고, 선생님이 수어와 자막으로 답해요.'],
};

export function render(root, { mode = 'word' }) {
  const c = state.curriculum;
  const [title, desc] = INTRO[mode] || INTRO.word;
  root.append(h('div.container',
    h('div.section-head', h('div', h('h2', { style: { fontSize: 'var(--fs-xl)' } }, title), h('p', { style: { maxWidth: '64ch', marginTop: '6px' } }, desc))),
    c?.offline ? h('div.card.card-pad', { style: { background: 'var(--warn-bg)', borderColor: 'transparent' } },
      '서버에 연결되지 않아 기본 과정만 보여요. 설정에서 서버 주소를 확인하세요.') : null,
    h('div.course-grid', mode === 'word' ? c.decks.map(deckCard)
      : mode === 'sentence' ? c.scenarios.map(scenarioCard)
        : c.topics.map(topicCard))));
}

function startBtn(mode, id) {
  return h('a.btn.btn-primary', { href: `#/session/${mode}/${id}` }, icon('play', 16), '시작');
}

function deckCard(d) {
  const ok = d.words.filter((w) => w.available).length;
  return h('div.card.course',
    h('div.meta', h('span.chip.chip-brand', `${ok}개 단어`)),
    h('h3', d.title),
    h('div.words', d.words.map((w) => h('span.chip', { dataset: w.available ? {} : { off: '' }, title: w.available ? '' : '인식 모델에 아직 없는 단어' }, w.gloss))),
    h('div.cta', h('span.muted', { style: { fontSize: '13px' } }, ok < d.words.length ? `${d.words.length - ok}개는 준비 중` : ''), startBtn('word', d.id)));
}

function scenarioCard(s) {
  return h('div.card.course',
    h('div.meta', h('span.chip.chip-accent', s.place), h('span.chip', `${s.turns.length}턴`)),
    h('h3', s.title),
    h('p.muted', { style: { fontSize: '14px' } }, s.goal),
    h('div.turns', s.turns.slice(0, 3).map((t) => h('div', t.partner))),
    h('div.cta', h('span'), startBtn('sentence', s.id)));
}

function topicCard(t) {
  return h('div.card.course',
    h('div.meta', h('span.chip', '숙련')),
    h('h3', t.title),
    h('p.muted', { style: { fontSize: '14px' } }, t.opener),
    h('div.cta', h('span'), startBtn('free_talk', t.id)));
}
