"""
WebSocket message schemas (Client ↔ Server).

Client → Server
  FrameMessage      : one sensor snapshot (camera + glove)
  StartSession      : begin a learning session
  EndSession        : end a learning session

Server → Client
  RecognitionResult : recognised sign + modal confidence
  LLMResponse       : tutor's natural-language reply + avatar commands
  FeedbackMessage   : inline correction during session
  SystemMessage     : info / warning / error notices
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

from .sensor import SensorFrame


# ─────────────────────── Client → Server ────────────────────────

class FrameMessage(BaseModel):
    type: Literal["frame"] = "frame"
    data: SensorFrame


class StartSession(BaseModel):
    type: Literal["start_session"] = "start_session"
    user_id: Optional[str] = None
    # 학습 모드: "word" 단어 연습 | "sentence" 상황별 문장 | "free_talk" 프리토킹
    mode: Literal["word", "sentence", "free_talk"] = "word"
    # word → 단어 묶음 id, sentence → 시나리오 id, free_talk → 주제 id (services/tutor/curriculum.py)
    scenario: str = "hospital"
    # word 모드에서 특정 단어만 연습할 때
    words: Optional[List[str]] = None


class TutorCommand(BaseModel):
    """학습 중 버튼 동작: 건너뛰기, 힌트, 다시 보기, (프리토킹) 말 보내기/지우기"""
    type: Literal["tutor_command"] = "tutor_command"
    action: Literal["skip", "hint", "repeat", "submit", "clear"]


class EndSession(BaseModel):
    type: Literal["end_session"] = "end_session"


# ─────────────────────── Server → Client ────────────────────────

class ModalityWeights(BaseModel):
    vision: float
    glove: float


class RecognitionResult(BaseModel):
    type: Literal["recognition"] = "recognition"
    text: str
    confidence: float
    modality_weights: ModalityWeights
    # True while the gesture window is still accumulating
    is_partial: bool = False


class AvatarCommand(BaseModel):
    # Animation clip name in the MetaHuman / UE5 motion library
    clip: str
    # Blend-in time in seconds
    blend_in: float = 0.1
    # Blend-out time in seconds
    blend_out: float = 0.1
    # Playback speed multiplier
    speed: float = 1.0
    # Optional facial expression override
    expression: Optional[str] = None


class LLMResponse(BaseModel):
    type: Literal["llm_response"] = "llm_response"
    # Korean natural-language text the tutor says
    text: str
    # Ordered sequence of avatar animation commands
    avatar_commands: List[AvatarCommand]
    # Tokens used — useful for monitoring API costs
    tokens_used: int = 0
    # prompt: 새 단어 안내 | correct: 정답 + 다음 단어 | feedback: 오답 교정 | reply: 프리토킹 답
    kind: Literal["prompt", "correct", "feedback", "reply", "hint"] = "prompt"
    # 튜터가 수어로 표현하는 글로스 순서 (아바타 클립과 자막에 사용)
    gloss: List[str] = []


class LessonTarget(BaseModel):
    gloss: List[str]                     # 표현해야 할 글로스 순서
    text: str                            # 자연스러운 한국어 문장
    description: Optional[str] = None    # 동작 설명 (단어 모드)
    clips: List[str] = []                # 아바타 시범 클립
    matched: int = 0                     # 지금까지 순서대로 맞춘 글로스 수


class LessonState(BaseModel):
    """현재 학습 단계 — 화면의 목표 카드, 진행률, 상대방 대사를 그린다."""
    type: Literal["lesson"] = "lesson"
    mode: str
    title: str
    step: int
    total: int
    target: Optional[LessonTarget] = None
    partner_line: Optional[str] = None   # 상황/프리토킹에서 상대(튜터)의 말
    correct: int = 0
    attempts: int = 0
    done: bool = False


class Utterance(BaseModel):
    """프리토킹: 사용자가 지금까지 수어로 말한 글로스 (보내기 전 버퍼)"""
    type: Literal["utterance"] = "utterance"
    gloss: List[str]
    final: bool = False
    thinking: bool = False   # 튜터가 답을 만드는 중 (Claude 응답 대기)


class InlineError(BaseModel):
    # Which body part the error concerns, e.g. "thumb", "wrist_angle"
    part: str
    description: str
    # Reference value (ground-truth) vs what the user did
    expected: Optional[Any] = None
    observed: Optional[Any] = None


class FeedbackMessage(BaseModel):
    type: Literal["feedback"] = "feedback"
    errors: List[InlineError]
    suggestions: List[str]
    # DTW distance to closest reference motion (lower = better)
    dtw_score: Optional[float] = None


class SessionSummary(BaseModel):
    type: Literal["session_summary"] = "session_summary"
    session_id: str
    total_signs: int
    correct_signs: int
    accuracy: float
    # Path to the generated PDF / JSON report
    report_url: Optional[str] = None
    # Per-sign breakdown
    errors_by_sign: Dict[str, List[InlineError]] = {}


class SystemMessage(BaseModel):
    type: Literal["system"] = "system"
    level: Literal["info", "warning", "error"] = "info"
    message: str


class CollectAck(BaseModel):
    type: Literal["collect_ack"] = "collect_ack"
    label: str
    count: int


class GloveStatus(BaseModel):
    type: Literal["glove_status"] = "glove_status"
    right: Optional[dict] = None   # GloveData dict or None
    left: Optional[dict] = None
