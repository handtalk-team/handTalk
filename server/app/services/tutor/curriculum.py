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
    # AI Hub 수어 3000단어(인식 모델 라벨)에 있는 단어만 쓴다
    {"id": "basics", "title": "기본 표현",
     "words": ["감사", "죄송", "괜찮다", "좋다", "싫다"]},
    {"id": "hospital", "title": "병원",
     "words": ["병원", "아프다", "의사", "간호사", "검사", "입원", "치료", "감기", "건강"]},
    {"id": "daily", "title": "일상",
     "words": ["밥", "친구", "가족"]},
    {"id": "emotions", "title": "감정",
     "words": ["즐겁다", "슬프다", "화나다", "무섭다", "피곤하다"]},
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
            {"partner": "어떻게 오셨어요?", "text": "아파서 왔어요.", "gloss": ["아프다"]},
            {"partner": "어디가 아프세요?", "text": "감기에 걸렸어요.", "gloss": ["감기"]},
            {"partner": "의사 선생님께 안내해 드릴게요.", "text": "감사합니다.", "gloss": ["감사"]},
        ],
    },
    {
        "id": "hospital_doctor",
        "title": "진료실",
        "place": "병원",
        "goal": "의사에게 증상을 설명하고 검사를 받아요.",
        "turns": [
            {"partner": "어떤 증상이 있으세요?", "text": "아파요. 감기 같아요.", "gloss": ["아프다", "감기"]},
            {"partner": "검사를 해 볼게요.", "text": "검사 괜찮아요.", "gloss": ["검사", "괜찮다"]},
            {"partner": "입원은 필요 없어요.", "text": "감사합니다.", "gloss": ["감사"]},
        ],
    },
    {
        "id": "admission",
        "title": "입원 상담",
        "place": "병원",
        "goal": "간호사와 입원과 치료 일정을 이야기해요.",
        "turns": [
            {"partner": "며칠 입원하셔야 해요.", "text": "입원 괜찮아요.", "gloss": ["입원", "괜찮다"]},
            {"partner": "가족분께 연락드릴까요?", "text": "가족 좋아요.", "gloss": ["가족", "좋다"]},
            {"partner": "치료 잘 받으세요.", "text": "감사합니다.", "gloss": ["감사"]},
        ],
    },
    {
        "id": "friend",
        "title": "친구와 안부",
        "place": "일상",
        "goal": "오랜만에 만난 친구와 안부를 나눠요.",
        "turns": [
            {"partner": "요즘 어떻게 지내?", "text": "피곤해.", "gloss": ["피곤하다"]},
            {"partner": "밥은 먹었어?", "text": "밥 좋아.", "gloss": ["밥", "좋다"]},
            {"partner": "건강 잘 챙겨!", "text": "고마워.", "gloss": ["감사"]},
        ],
    },
]

FREE_TALK_TOPICS: List[dict] = [
    {"id": "daily", "title": "오늘 하루", "opener": "오늘 하루는 어땠어요? 기분이 어떤지 수어로 말해 주세요."},
    {"id": "health", "title": "건강", "opener": "요즘 몸은 괜찮으세요? 아픈 데는 없어요?"},
    {"id": "free", "title": "자유 주제", "opener": "무엇이든 편하게 이야기해 주세요. 수어로 천천히 말해도 괜찮아요."},
]

# ─────────────────────── 단어 설명 ─────────────────────────────────
# 알려진 단어는 사전 설명을 그대로 쓴다 (소형 LLM 은 한국 수어 지식이 부족함)

# 동작 설명: 국립국어원 한국수어사전(sldict.korean.go.kr)의 수형 설명을 쉬운 말로 옮긴 것.
# 뜻이 같은 수어가 여럿이면 AI Hub 단어 영상(아바타가 배운 동작)과 손 위치·사용 손이 맞는 쪽을 골랐다.
KSL_DESCRIPTIONS: Dict[str, str] = {
    "감사": "왼손을 손등이 위로 가게 펴고, 오른손을 세워 새끼손가락 쪽 옆면으로 왼 손등을 두 번 톡톡 댑니다.",
    "죄송": "오른손 엄지와 검지로 동그라미를 만들어 이마에 댔다가, 손가락을 펴면서 왼 손등 위에 내려놓습니다.",
    "괜찮다": "오른손 새끼손가락만 펴서 손가락 끝을 턱에 가볍게 두 번 댑니다.",
    "좋다": "오른손 엄지를 세워 몸 앞에서 앞뒤로 두 번 움직입니다.",
    "싫다": "오른손 엄지와 검지를 펴서 벌리고, 손가락 끝으로 가슴 가운데를 두 번 툭툭 칩니다.",
    "병원": "왼손을 손등이 앞으로 가게 펴고, 오른손 검지·중지를 펴서 손가락 끝으로 왼 손등을 두 번 두드립니다 (진찰받는 모습).",
    "아프다": "오른손을 손바닥이 위로 가게 약간 오므리고 좌우로 흔듭니다.",
    "의사": "오른손 검지·중지로 왼 손등을 두 번 두드린 다음, 검지 옆면으로 왼 주먹 손목을 두 번 두드립니다 (맥 짚기).",
    "간호사": "오른손 검지로 이를 가리키고, 두 검지로 '+'자를 만들어 이마에 댄 다음, 두 손의 엄지·새끼손가락을 펴서 맞대고 좌우로 살짝 돌리며 벌립니다.",
    "검사": "오른손 검지·중지를 약간 구부려 두 눈을 향하게 하고 왼쪽으로 돌린 다음, 두 손 엄지를 세워 동시에 아래로 내립니다.",
    "입원": "오른손 검지·중지로 왼 손등을 두 번 두드린 다음, 손바닥이 위로 가게 편 왼손 위에 오른손 검지·중지를 눕혀 올립니다.",
    "치료": "오른손을 세워 엄지 쪽 옆면을 이마 가운데에 댄 다음, 검지를 편 두 주먹을 가운데로 모아 두 팔을 엇갈리게 합니다.",
    "감기": "오른손 엄지와 검지로 동그라미를 만들어 턱 밑에 댔다가 입을 벌리며 내민 뒤, 검지·중지로 코밑을 두 번 스쳐 내립니다.",
    "건강": "두 주먹을 가슴 앞에서 힘주어 몸 쪽으로 당깁니다.",
    "밥": "오른손을 손바닥이 위로 가게 펴서 숟가락처럼 입 쪽으로 두 번 올립니다.",
    "친구": "두 손을 두 번 맞잡습니다.",
    "가족": "두 손끝을 맞대 지붕 모양을 만든 다음, 왼손은 그대로 두고 오른손 새끼·엄지를 펴서 흔들며 오른쪽으로 옮깁니다.",
    "즐겁다": "두 손을 약간 구부려 손끝을 양쪽 가슴에 대고 위아래로 엇갈리게 두 번 움직입니다.",
    "슬프다": "오른손 엄지와 검지 끝을 맞대 오른쪽 눈 밑에 댔다가 내리고, 손을 펴서 구부려 가슴에 대고 돌립니다.",
    "화나다": "오른손 엄지를 세워 배에서 위로 힘주어 올립니다.",
    "무섭다": "두 손 엄지 끝을 몸 가운데의 위와 아래에 대고 손가락을 떱니다.",
    "피곤하다": "두 손바닥을 양쪽 가슴에 댔다가 힘없이 밑으로 내립니다.",
}

# 아바타 클립 이름 (Unreal /Game/HandTalk/Signs, 웹 미리보기 motions/*.json 과 같음)
GLOSS_TO_CLIP: Dict[str, str] = {
    "병원": "KSL_hospital", "아프다": "KSL_hurt", "의사": "KSL_doctor", "괜찮다": "KSL_okay",
    "감사": "KSL_thank_you", "간호사": "KSL_nurse", "검사": "KSL_exam", "입원": "KSL_admission",
    "치료": "KSL_treatment", "감기": "KSL_cold", "건강": "KSL_health", "밥": "KSL_meal",
    "친구": "KSL_friend", "가족": "KSL_family", "좋다": "KSL_good", "싫다": "KSL_dislike",
    "슬프다": "KSL_sad", "화나다": "KSL_angry", "무섭다": "KSL_scared", "피곤하다": "KSL_tired",
    "죄송": "KSL_sorry", "즐겁다": "KSL_happy",
}


# 전체 단어 사전 (AI Hub 3000단어 모션을 가져오면 생김: client/unreal/.../Scripts/import_motion_zip.py)
# AI Hub 데이터에서 나온 목록이라 저장소에 없고, 없으면 위 GLOSS_TO_CLIP 만 쓴다.
LEXICON_PATH = os.environ.get(
    "HANDTALK_LEXICON",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "client", "assets", "lexicon.json"))


@lru_cache(maxsize=1)
def lexicon() -> Dict[str, str]:
    """글로스 → 아바타 클립 (사전 + 직접 지정한 이름)"""
    out: Dict[str, str] = {}
    try:
        import json
        with open(LEXICON_PATH, encoding="utf-8") as f:
            out = {e["gloss"]: e["clip"] for e in json.load(f)}
    except (OSError, ValueError, KeyError):
        pass
    out.update(GLOSS_TO_CLIP)
    return out


def clip_for(gloss: str) -> Optional[str]:
    return lexicon().get(gloss)


SIGN_MODEL_META = os.path.join(os.environ.get("HANDTALK_SIGN_MODEL", r"C:\handtalk\models\sign_motion"), "meta.json")


@lru_cache(maxsize=1)
def sign_vocab() -> frozenset:
    """선생님이 수어로 할 수 있는 단어 = 동작 생성 모델이 배운 단어 (AI Hub 단어 + 문장 단어)"""
    try:
        import json
        with open(SIGN_MODEL_META, encoding="utf-8") as f:
            return frozenset(json.load(f)["vocab"])
    except (OSError, ValueError, KeyError):
        return frozenset(lexicon())   # 모델이 아직 없을 때 (단어 목록만)


def can_sign(gloss: str) -> bool:
    return gloss in sign_vocab()


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
        "clip": clip_for(word),
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
        "avatar_signs": len(lexicon()),
    }
