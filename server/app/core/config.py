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

    # ── LLM Provider ─────────────────────────────────────────────────────────
    # "ollama"  : 완전 무료, 로컬 실행 (기본값)
    # "groq"    : 무료 클라우드 API (GROQ_API_KEY 필요, 속도 매우 빠름)
    # "claude"  : Anthropic Claude (ANTHROPIC_API_KEY 필요, 유료)
    LLM_PROVIDER: str = "ollama"

    # Ollama 설정 (로컬 무료)
    OLLAMA_BASE_URL: str = "http://localhost:11434/v1"
    OLLAMA_MODEL: str = "llama3.2"           # ollama run llama3.2

    # Groq 설정 (무료 클라우드, console.groq.com에서 발급)
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"  # 또는 llama-3.3-70b-versatile

    # Anthropic 설정 (유료, 선택사항)
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-sonnet-4-6"

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

    # ── Glove (USB Serial) ────────────────────────────────────────────────────
    # Serial port for the ESP32 right-hand glove.  Leave empty to disable.
    # macOS example: /dev/cu.usbserial-0001
    # Linux example: /dev/ttyUSB0
    GLOVE_PORT_RIGHT: str = ""
    GLOVE_PORT_LEFT: str = ""
    GLOVE_BAUD_RATE: int = 115200

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
