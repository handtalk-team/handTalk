import { h, icon } from '../core/dom.js';
import { state, loadSettings } from '../core/store.js';
import { summary, loadHistory } from '../core/history.js';
import { modeTitle } from '../core/api.js';

const MODE_ICON = { word: 'word', sentence: 'sentence', free_talk: 'chat' };

function fmtDate(iso) {
  const d = new Date(iso);
  return `${d.getMonth() + 1}월 ${d.getDate()}일 ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

export function render(root) {
  const s = summary();
  const last = loadHistory()[0];
  const name = loadSettings().name;
  const cur = state.curriculum;
  const firstDeck = cur?.decks?.[0]?.id || 'hospital';

  const resume = last
    ? h('a.btn.btn-accent.btn-lg', { href: `#/session/${last.mode}/${last.scenario}` }, icon('play'), `이어서 하기 · ${last.title}`)
    : h('a.btn.btn-accent.btn-lg', { href: `#/session/word/${firstDeck}` }, icon('play'), '첫 수업 시작하기');

  root.append(h('div.container',
    h('section.hero',
      h('div.hero-main',
        h('div.eyebrow', { style: { color: 'rgba(255,255,255,.7)' } }, 'AI 수어 튜터'),
        h('h2', name ? `${name}님, 오늘도 손으로 대화해 볼까요?` : '손으로 말하고, 선생님과 바로 대화해요'),
        h('p', '카메라와 장갑으로 수어를 하면 바로 인식하고, 아바타 선생님이 수어와 자막으로 답해 드려요. 단어부터 상황별 문장, 자유 대화까지 단계별로 연습하세요.'),
        h('div.actions', resume, h('a.btn.btn-lg', { href: '#/learn/free_talk' }, icon('chat'), '프리토킹'))),
      h('div.stats',
        stat(s.streak, '일', '연속 학습'),
        stat(s.accuracy ?? '—', s.accuracy == null ? '' : '%', '평균 인식 정확도'),
        stat(s.minutes, '분', '누적 학습 시간'),
        stat(s.words, '개', '익힌 단어'))),

    h('section',
      h('div.section-head', h('div', h('h2', '학습 모드'), h('p', '실력에 맞는 모드를 골라 시작하세요.'))),
      h('div.mode-grid', { style: { marginTop: '14px' } },
        (cur?.modes || []).map((m) => h('a.card.mode-card', { href: `#/learn/${m.id}`, dataset: { mode: m.id } },
          h('div.ico', icon(MODE_ICON[m.id] || 'word', 22)),
          h('div', h('span.chip', m.level), ' '),
          h('h3', m.title),
          h('p', m.summary),
          h('div.foot', h('span', courseCount(m.id)), h('span', { style: { color: 'var(--brand-600)', display: 'inline-flex', gap: '4px', alignItems: 'center', fontWeight: 600 } }, '시작', icon('arrow', 16))))))),

    h('section',
      h('div.section-head', h('div', h('h2', '최근 학습')), h('a.btn.btn-ghost.btn-sm', { href: '#/history' }, '전체 기록', icon('arrow', 14))),
      h('div.card', { style: { marginTop: '14px' } }, recent()))));
}

function stat(v, unit, label) {
  return h('div.card.stat', h('div.v.num', String(v), unit ? h('small', unit) : null), h('div.l', label));
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
    h('span.chip.chip-brand', modeTitle(r.mode)),
    h('div.grow', h('div.t', r.title), h('div.d', fmtDate(r.date))),
    h('div.d.num', r.attempts ? `${r.correct}/${r.attempts}` : ''),
    h('a.btn.btn-sm', { href: `#/session/${r.mode}/${r.scenario}` }, '다시 하기'))));
}
