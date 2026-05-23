from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import relationship

from .user import Base


class DataSample(Base):
    __tablename__ = "data_samples"

    id         = Column(String, primary_key=True)
    label      = Column(String, nullable=False, index=True)
    file_path  = Column(String, nullable=False)
    frames     = Column(Integer, nullable=False)   # T (시퀀스 길이)
    collected_at = Column(DateTime, default=datetime.utcnow)


class LearningSession(Base):
    __tablename__ = "learning_sessions"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    scenario = Column(String, default="free_talk")
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)

    total_signs = Column(Integer, default=0)
    correct_signs = Column(Integer, default=0)
    # Overall recognition accuracy for this session
    accuracy = Column(Float, default=0.0)

    # Raw per-sign feedback stored as JSON list
    sign_log = Column(JSON, default=list)
    # Aggregated error map  {sign_label: [error_dicts]}
    errors_by_sign = Column(JSON, default=dict)

    user = relationship("User", backref="sessions", foreign_keys=[user_id])


class SignAttempt(Base):
    __tablename__ = "sign_attempts"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("learning_sessions.id"), nullable=False)
    sign_label = Column(String, nullable=False)
    is_correct = Column(Integer, default=0)          # 0 / 1
    confidence = Column(Float, default=0.0)
    dtw_score = Column(Float, nullable=True)
    # Full fused feature window stored for 3-D replay
    feature_window = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
