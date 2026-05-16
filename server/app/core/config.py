"""
Central configuration — all env-var driven so values never live in code.
Copy server/.env.example → server/.env and fill in the blanks.
"""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── App ─────────────────────────────────────────────────────────────────
    APP_NAME: str = "handTalk API"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── Database ─────────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./handtalk.db"

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── JWT ───────────────────────────────────────────────────────────────────
    JWT_SECRET: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60 * 24 * 7     # 1 week

    # ── Anthropic (Claude) ───────────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = ""
    LLM_MODEL: str = "claude-sonnet-4-6"
    # Maximum tokens for LLM reply — keep tight to stay under latency budget
    LLM_MAX_TOKENS: int = 512

    # ── Recognition ──────────────────────────────────────────────────────────
    # Path to the ONNX model file.  If missing, the rule-based fallback is used.
    ONNX_MODEL_PATH: str = "ml/models/sign_recognizer.onnx"
    # Sliding window length (frames) fed to the recogniser
    WINDOW_SIZE: int = 60
    # Minimum softmax probability to accept a recognition result
    RECOGNITION_THRESHOLD: float = 0.70
    # Minimum number of frames that must be in the window before inference runs
    MIN_WINDOW_FRAMES: int = 30

    # ── Sensor quality thresholds ─────────────────────────────────────────────
    # Below these values the modal is down-weighted in fusion
    VISION_CONFIDENCE_MIN: float = 0.30
    GLOVE_QUALITY_MIN: float = 0.40

    # ── Camera ────────────────────────────────────────────────────────────────
    # OpenCV camera index for local webcam (0 = built-in laptop cam)
    CAMERA_INDEX: int = 0
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480
    CAMERA_FPS: int = 30

    # ── Mock glove ────────────────────────────────────────────────────────────
    # Hz at which the mock glove generates packets
    MOCK_GLOVE_HZ: int = 50
    # Whether to use the mock glove when no BLE device is connected
    USE_MOCK_GLOVE: bool = True

    # ── CORS ──────────────────────────────────────────────────────────────────
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
