"""
한국수어(KSL) 문장 구성 규칙과 실제 수어 문장 예시 — 튜터가 답을 수어 어순(글로스)으로 짤 때 Claude 에게 준다.

규칙 근거
  - 한국수어 문법 개관 (기본 SOV, 수식어가 앞, 조사·어미 없음, 의문사는 문장 끝, 시제는 시간어/완료 표지,
    부정은 부정 어휘를 서술어 뒤에)  https://namu.wiki/w/한국%20수어
  - 수어 격 실현은 어순과 서술어 결합으로 (국어·수어 비교 연구)  https://koreascience.kr/article/CFKO201335553772468.pdf
  - AI Hub 수어 문장 2,000개의 실제 글로스 순서 통계 (이 프로젝트에서 집계):
    의문사 무엇·어디·방법 등은 대부분 문장 끝, 가능/되다/맞다/원하다 같은 서술 표지는 끝, 대명사는 거의 생략.

예시 문장은 AI Hub 데이터에서 나온 것이라 저장소에 두지 않는다 (HANDTALK_SENTENCES, 기본 C:\\handtalk\\sentdata\\labels\\sentences.json).
"""

from __future__ import annotations

import json
import os
import random
from functools import lru_cache
from typing import Iterable, List

from app.core.config import local_or

SENTENCES_PATH = os.environ.get("HANDTALK_SENTENCES") or local_or("ksl_sentences.json", r"C:\handtalk\sentdata\labels\sentences.json")

KSL_RULES = """[한국수어 문장 규칙 — gloss 는 반드시 이 규칙으로 짠다]
1. 조사·어미를 쓰지 않는다. 은/는/이/가/을/를/에/에서/요/습니다 같은 말은 gloss 에 넣지 않고 기본형 단어만 쓴다. (예: "병원에 갔어요" → 병원 가다 끝)
2. 기본 어순은 주어 → 목적어 → 서술어. 꾸미는 말은 꾸밈 받는 말 앞에 온다. (예: 빨간 가방 → 빨갛다 가방)
3. 장소·주제를 먼저 세우고 그에 대해 말한다. (예: "지하철역에 카페 있어요?" → 지하철 곳 카페 있다)
4. 시간은 문장 맨 앞에 둔다. 과거·완료는 시간어나 서술어 뒤의 '끝/완료'로, 미래는 '내일/나중' 같은 시간어로 나타낸다.
5. 의문사(무엇, 어디, 언제, 누구, 왜, 어떻게, 몇, 얼마)는 문장 끝에 둔다. 장소를 묻는 말은 '곳'이나 '어디', 방법을 묻는 말은 '방법'으로 끝낸다. (예: "어디서 갈아타요?" → 갈아타다 곳 어디)
6. 예/아니오 질문은 평서문 어순 그대로 두고 끝에 '맞다'를 붙일 수 있다. (예: "이 버스 시청 가요?" → 버스 시청 가다 맞다)
7. 부정·불가능은 서술어 뒤에 둔다: 없다, 아니다, 못하다, 안되다, 불가능, 모르다, 싫다. (예: "카드가 안 돼요" → 카드 사용 안되다)
8. 바람·가능·필요도 서술어 뒤에: 원하다, 가능, 되다, 필요. (예: "표를 사고 싶어요" → 표 사다 원하다)
9. 나/너 같은 대명사는 뜻이 분명하면 생략한다.
10. 숫자, 고유명사(역 이름 등)는 단어 그대로 쓰되 목록에 있는 것만 쓴다."""


@lru_cache(maxsize=1)
def sentences() -> List[List[str]]:
    try:
        with open(SENTENCES_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    seen, out = set(), []
    for s in data:
        seq = s.get("gloss_sequence") or []
        key = " ".join(seq)
        if len(seq) >= 2 and key not in seen:
            seen.add(key)
            out.append(seq)
    return out


def examples(context_words: Iterable[str], n: int = 24) -> List[List[str]]:
    """대화에 나온 단어와 겹치는 실제 수어 문장을 우선, 나머지는 의문문·부정문이 섞이게 고른다."""
    pool = sentences()
    if not pool:
        return []
    words = set(context_words)
    scored = sorted(pool, key=lambda q: -len(words & set(q)))
    picked = [q for q in scored[: n // 2] if words & set(q)]
    rng = random.Random(len(words))
    rest = [q for q in pool if q not in picked]
    rng.shuffle(rest)
    # 문장 끝이 의문사·부정·바람인 예시를 일부러 섞는다
    enders = {"무엇", "어디", "곳", "방법", "맞다", "안되다", "없다", "불가능", "원하다", "가능", "언제", "누구", "왜"}
    picked += [q for q in rest if q[-1] in enders][: (n - len(picked)) // 2]
    picked += [q for q in rest if q not in picked][: n - len(picked)]
    return picked[:n]


def prompt_block(context_words: Iterable[str]) -> str:
    ex = examples(context_words)
    lines = [KSL_RULES]
    if ex:
        lines.append("\n[실제 한국수어 문장의 단어 순서 예시 (AI Hub 수어 데이터)]")
        lines += ["- " + " ".join(q) for q in ex]
    return "\n".join(lines)
