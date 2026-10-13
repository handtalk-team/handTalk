/** 학습 기록 — 이 브라우저에만 저장 (서버 DB 의 세션 기록과 별개인 개인 대시보드용) */

const KEY = 'handtalk.history.v1';

export function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(KEY) || '[]');
  } catch {
    return [];
  }
}

export function addRecord(rec) {
  const list = loadHistory();
  list.unshift({ id: Date.now(), ...rec });
  try { localStorage.setItem(KEY, JSON.stringify(list.slice(0, 300))); } catch { /* 저장 불가 */ }
}

export function summary() {
  const list = loadHistory();
  const days = new Set(list.map((r) => r.date.slice(0, 10)));
  let streak = 0;
  const d = new Date();
  for (;;) {
    const key = d.toISOString().slice(0, 10);
    if (!days.has(key)) break;
    streak += 1;
    d.setDate(d.getDate() - 1);
  }
  const attempts = list.reduce((a, r) => a + (r.attempts || 0), 0);
  const correct = list.reduce((a, r) => a + (r.correct || 0), 0);
  const minutes = Math.round(list.reduce((a, r) => a + (r.seconds || 0), 0) / 60);
  return {
    sessions: list.length,
    streak,
    accuracy: attempts ? Math.round((correct / attempts) * 100) : null,
    minutes,
    words: new Set(list.flatMap((r) => r.learned || [])).size,
  };
}

/** 최근 14일 일별 학습 시간(분) */
export function daily(daysBack = 14) {
  const list = loadHistory();
  const out = [];
  for (let i = daysBack - 1; i >= 0; i--) {
    const d = new Date();
    d.setDate(d.getDate() - i);
    const key = d.toISOString().slice(0, 10);
    const mins = list.filter((r) => r.date.startsWith(key)).reduce((a, r) => a + (r.seconds || 0), 0) / 60;
    out.push({ key, label: `${d.getMonth() + 1}/${d.getDate()}`, minutes: Math.round(mins), today: i === 0 });
  }
  return out;
}
