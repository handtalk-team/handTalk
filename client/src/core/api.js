import { serverBase, state, emit } from './store.js';

/** 서버가 꺼져 있어도 화면을 둘러볼 수 있게 하는 최소 과정 (서버 curriculum.py 와 같은 구조) */
const FALLBACK = {
  modes: [
    { id: 'word', title: '단어 연습', level: '입문', summary: '아바타 선생님의 시범을 보고 단어 하나씩 따라 해요.' },
    { id: 'sentence', title: '상황별 문장', level: '중급', summary: '실제 상황에서 문장을 수어 순서대로 표현해요.' },
    { id: 'free_talk', title: '프리토킹', level: '숙련', summary: '튜터와 수어로 자유롭게 대화해요.' },
  ],
  decks: [
    { id: 'hospital', title: '병원', words: ['병원', '아프다', '의사', '간호사', '검사'].map((g) => ({ gloss: g, available: true })) },
  ],
  scenarios: [
    { id: 'hospital_reception', title: '병원 접수처', place: '병원', goal: '접수처에서 증상을 말하고 진료를 받아요.',
      turns: [{ partner: '어떻게 오셨어요?', text: '아파서 왔어요.', gloss: ['아프다'], available: true }] },
  ],
  topics: [{ id: 'daily', title: '오늘 하루', opener: '오늘 하루는 어땠어요?' }],
  vocab_size: 0,
  offline: true,
};

export async function loadCurriculum() {
  try {
    const res = await fetch(`${serverBase()}/api/v1/curriculum`, { cache: 'no-store' });
    if (!res.ok) throw new Error(res.status);
    state.curriculum = await res.json();
  } catch (e) {
    console.warn('[api] curriculum 불러오기 실패, 기본 과정 사용', e);
    state.curriculum = FALLBACK;
  }
  emit('curriculum', state.curriculum);
  return state.curriculum;
}

export function modeTitle(mode) {
  return (state.curriculum?.modes || FALLBACK.modes).find((m) => m.id === mode)?.title || mode;
}

export function courseTitle(mode, id) {
  const c = state.curriculum || FALLBACK;
  const list = mode === 'word' ? c.decks : mode === 'sentence' ? c.scenarios : c.topics;
  return list.find((x) => x.id === id)?.title || id;
}
