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
    # Scenario narrows the vocabulary used for recognition & LLM prompting
    # e.g. "free_talk" | "greetings" | "numbers" | "emotions"
    scenario: str = "free_talk"


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
    # prompt: 새 단어 안내 | correct: 정답 + 다음 단어 | feedback: 오답 교정
    kind: Literal["prompt", "correct", "feedback"] = "prompt"


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
