import { h } from '../core/dom.js';
import { summary, loadHistory, daily } from '../core/history.js';
import { modeTitle } from '../core/api.js';

export function render(root) {
  const s = summary();
  const days = daily(14);
  const max = Math.max(10, ...days.map((d) => d.minutes));
  const list = loadHistory();

  root.append(h('div.container',
    h('div.section-head', h('div', h('h2', { style: { fontSize: 'var(--fs-xl)' } }, '학습 기록'),
      h('p', '이 브라우저에서 진행한 수업이 저장돼요.'))),
    h('div.stats', { style: { gridTemplateColumns: 'repeat(4, minmax(0, 1fr))' } },
      card(s.sessions, '회', '총 수업'), card(s.minutes, '분', '누적 학습 시간'),
      card(s.accuracy ?? '—', s.accuracy == null ? '' : '%', '평균 정확도'), card(s.streak, '일', '연속 학습')),
    h('div.card.card-pad',
      h('div.section-head', h('h2', '최근 2주 학습 시간'), h('span.muted.num', { style: { fontSize: '13px' } }, `최대 ${max}분 기준`)),
      h('div.bars', { role: 'img', 'aria-label': '최근 14일 일별 학습 시간' },
        days.map((d) => h('div.b', { title: `${d.label} ${d.minutes}분`, dataset: d.today ? { today: '' } : {}, style: { height: `${(d.minutes / max) * 100}%` } }))),
      h('div.bars-x', days.map((d, i) => h('span', i % 2 ? '' : d.label)))),
    h('div.card', list.length ? h('div.list', list.slice(0, 50).map((r) => h('div.list-row',
      h('span.chip.chip-brand', modeTitle(r.mode)),
      h('div.grow', h('div.t', r.title), h('div.d', new Date(r.date).toLocaleString('ko-KR'))),
      h('div.d.num', `${Math.max(1, Math.round((r.seconds || 0) / 60))}분`),
      h('div.d.num', { style: { width: '70px', textAlign: 'right' } }, r.attempts ? `${r.correct}/${r.attempts}` : '—'))))
      : h('div.empty', '아직 기록이 없어요.'))));
}

function card(v, unit, label) {
  return h('div.card.stat', h('div.v.num', String(v), unit ? h('small', unit) : null), h('div.l', label));
}
