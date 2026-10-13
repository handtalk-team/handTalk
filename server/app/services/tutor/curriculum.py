"""
학습 과정(커리큘럼)
==================

세 가지 학습 모드가 같은 글로스(수어 단어) 사전을 쓴다.

  word      단어 연습     — 단어 하나씩 시범을 보고 따라 하기
  sentence  상황별 문장   — 병원, 식당 같은 상황에서 정해진 문장(글로스 순서)을 표현
  free_talk 프리토킹      — 튜터와 자유롭게 대화 (숙련자)

단어는 인식 모델의 labels.txt 에 있는 것만 실제로 연습할 수 있다. 시나리오의 단어 중 모델이
모르는 단어는 available=False 로 표시해 UI 에서 회색으로 보인다.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Optional

from app.core.config import get_settings

settings = get_settings()

# ─────────────────────── 모드 ───────────────────────────────────────

MODES: List[dict] = [
    {
        "id": "word",
        "title": "단어 연습",
        "level": "입문",
        "summary": "아바타 선생님의 시범을 보고 단어 하나씩 따라 해요. 틀리면 손 모양과 움직임을 바로 알려 드려요.",
    },
    {
        "id": "sentence",
        "title": "상황별 문장",
        "level": "중급",
        "summary": "병원, 식당, 길 찾기 같은 실제 상황에서 문장을 수어 순서대로 표현해요.",
    },
    {
        "id": "free_talk",
        "title": "프리토킹",
        "level": "숙련",
        "summary": "주제 없이 튜터와 수어로 자유롭게 대화해요. 튜터가 수어와 자막으로 답해요.",
    },
]

# ─────────────────────── 단어 묶음 ─────────────────────────────────

WORD_DECKS: List[dict] = [
    {"id": "greetings", "title": "인사와 기본 표현",
     "words": ["안녕하세요", "감사합니다", "감사", "미안하다", "괜찮다", "반갑다", "네", "아니다"]},
    {"id": "hospital", "title": "병원",
     "words": ["병원", "아프다", "의사", "간호사", "약", "검사", "입원", "치료", "감기", "건강"]},
    {"id": "daily", "title": "일상",
     "words": ["물", "밥", "화장실", "집", "학교", "친구", "가족", "오늘", "내일"]},
    {"id": "emotions", "title": "감정",
     "words": ["좋다", "싫다", "슬프다", "화나다", "무섭다", "기쁘다", "피곤하다"]},
]

# ─────────────────────── 상황별 문장 ───────────────────────────────
# gloss: 수어 어순의 단어 목록 (인식 결과를 이 순서로 맞추면 정답)
# partner: 상황 속 상대방(튜터가 연기)의 말

SCENARIOS: List[dict] = [
    {
        "id": "hospital_reception",
        "title": "병원 접수처",
        "place": "병원",
        "goal": "접수처에서 증상을 말하고 진료를 받아요.",
        "turns": [
            {"partner": "안녕하세요, 어떻게 오셨어요?", "text": "안녕하세요, 아파서 왔어요.",
             "gloss": ["안녕하세요", "아프다"]},
            {"partner": "어디가 아프세요?", "text": "감기에 걸렸어요.",
             "gloss": ["감기"]},
            {"partner": "의사 선생님께 안내해 드릴게요.", "text": "감사합니다.",
             "gloss": ["감사합니다"]},
        ],
    },
    {
        "id": "pharmacy",
        "title": "약국",
        "place": "약국",
        "goal": "처방받은 약을 받고 복용법을 물어봐요.",
        "turns": [
            {"partner": "처방전 주시겠어요?", "text": "네, 약 주세요.",
             "gloss": ["네", "약"]},
            {"partner": "하루 세 번 드세요.", "text": "괜찮아요, 감사합니다.",
             "gloss": ["괜찮다", "감사합니다"]},
        ],
    },
    {
        "id": "hospital_doctor",
        "title": "진료실",
        "place": "병원",
        "goal": "의사에게 증상을 설명하고 검사를 받아요.",
        "turns": [
            {"partner": "어떤 증상이 있으세요?", "text": "아파요. 감기 같아요.",
             "gloss": ["아프다", "감기"]},
            {"partner": "검사를 해 볼게요.", "text": "검사 괜찮아요.",
             "gloss": ["검사", "괜찮다"]},
            {"partner": "입원은 필요 없어요.", "text": "감사합니다.",
             "gloss": ["감사합니다"]},
        ],
    },
    {
        "id": "first_meeting",
        "title": "처음 만난 사람",
        "place": "일상",
        "goal": "처음 만난 사람과 인사하고 자기소개를 해요.",
        "turns": [
            {"partner": "안녕하세요! 처음 뵙겠습니다.", "text": "안녕하세요, 반가워요.",
             "gloss": ["안녕하세요", "반갑다"]},
            {"partner": "오늘 날씨 좋네요.", "text": "네, 좋아요.",
             "gloss": ["네", "좋다"]},
        ],
    },
]

FREE_TALK_TOPICS: List[dict] = [
    {"id": "daily", "title": "오늘 하루", "opener": "안녕하세요! 오늘 하루는 어땠어요?"},
    {"id": "health", "title": "건강", "opener": "요즘 몸은 괜찮으세요? 아픈 데는 없어요?"},
    {"id": "free", "title": "자유 주제", "opener": "무엇이든 편하게 이야기해 주세요. 수어로 천천히 말해도 괜찮아요."},
]

# ─────────────────────── 단어 설명 ─────────────────────────────────
# 알려진 단어는 사전 설명을 그대로 쓴다 (소형 LLM 은 한국 수어 지식이 부족함)

KSL_DESCRIPTIONS: Dict[str, str] = {
    "안녕하세요": "오른손을 활짝 펴 손바닥이 상대방을 향하도록 이마 옆에 가볍게 댄 뒤, 앞으로 살짝 내밀며 인사합니다.",
    "아프다": "양손 검지를 세워 서로 마주보게 한 후, 번갈아 가며 위아래로 두 번 반복해서 움직입니다.",
    "병원": "왼손 등 위에 오른손 검지와 중지를 교차하여 십자(+) 모양을 만듭니다.",
    "도와주세요": "한 손바닥 위에 반대 손 주먹을 올린 뒤, 두 손을 함께 위로 들어올립니다.",
    "감사합니다": "오른손을 펴서 손끝을 턱 아래에 살짝 댄 뒤, 앞으로 내밀며 고마움을 표현합니다.",
    "감사": "오른손을 펴서 손끝을 턱 아래에 살짝 댄 뒤, 앞으로 내밀며 고마움을 표현합니다.",
    "약": "왼손바닥 위에 오른손 중지 끝을 대고 작은 원을 두 번 그립니다.",
    "의사": "오른손 집게손가락으로 왼손 손목 안쪽(맥박 부위)을 두세 번 두드립니다.",
    "간호사": "오른손 집게손가락과 중지를 펴서 왼손 팔목 위에 두 번 얹습니다.",
    "괜찮아요": "오른손 엄지를 세운 뒤 가볍게 앞으로 두 번 내밉니다.",
    "괜찮다": "오른손 엄지를 세운 뒤 가볍게 앞으로 두 번 내밉니다.",
    "물": "오른손 W 모양(검지·중지·약지를 펴고 나머지는 접음)으로 입 앞에서 아래로 내립니다.",
    "화장실": "오른손 T 모양(엄지를 검지와 중지 사이에 끼움)으로 손목을 좌우로 흔듭니다.",
}

# 아바타 클립 이름 (Unreal /Game/HandTalk/Signs, 웹 미리보기 motions/*.json 과 같음)
GLOSS_TO_CLIP: Dict[str, str] = {
    "병원": "KSL_hospital",
    "아프다": "KSL_hurt",
    "의사": "KSL_doctor",
    "괜찮다": "KSL_okay",
    "괜찮아요": "KSL_okay",
    "감사": "KSL_thank_you",
    "감사합니다": "KSL_thank_you",
}


# ─────────────────────── 조회 ───────────────────────────────────────

@lru_cache(maxsize=1)
def model_vocab() -> frozenset:
    """인식 모델이 아는 단어 (labels.txt). 없으면 빈 집합 → 모든 단어를 사용 가능으로 본다."""
    path = os.path.join(os.path.dirname(settings.ONNX_MODEL_PATH), "labels.txt")
    if not os.path.exists(path):
        return frozenset()
    with open(path, encoding="utf-8") as f:
        return frozenset(line.strip() for line in f if line.strip())


def is_available(word: str) -> bool:
    vocab = model_vocab()
    return not vocab or word in vocab


def word_entry(word: str) -> dict:
    return {
        "gloss": word,
        "available": is_available(word),
        "description": KSL_DESCRIPTIONS.get(word),
        "clip": GLOSS_TO_CLIP.get(word),
    }


def deck(deck_id: str) -> Optional[dict]:
    return next((d for d in WORD_DECKS if d["id"] == deck_id), None)


def scenario(scenario_id: str) -> Optional[dict]:
    return next((s for s in SCENARIOS if s["id"] == scenario_id), None)


def topic(topic_id: str) -> Optional[dict]:
    return next((t for t in FREE_TALK_TOPICS if t["id"] == topic_id), None)


def catalog() -> dict:
    """웹 앱이 화면을 그릴 때 쓰는 전체 목록 (GET /api/v1/curriculum)"""
    vocab = model_vocab()
    return {
        "modes": MODES,
        "decks": [
            {**d, "words": [word_entry(w) for w in d["words"]]}
            for d in WORD_DECKS
        ],
        "scenarios": [
            {**s, "turns": [{**t, "available": all(is_available(g) for g in t["gloss"])}
                            for t in s["turns"]]}
            for s in SCENARIOS
        ],
        "topics": FREE_TALK_TOPICS,
        "vocab_size": len(vocab),
    }
