"""
handTalk — FastAPI Application Entry Point
==========================================

Endpoints
---------
GET  /healthz                   — liveness probe
WS   /ws                        — real-time sign language session
WS   /ws/avatar                 — Unreal MetaHuman client (receives tutor replies)
GET  /sessions/{id}             — session metadata (REST)
GET  /feedback/{id}/summary     — post-session summary (REST)
GET  /feedback/{id}/error-note  — auto-generated error note (REST)
GET  /curriculum                — learning modes, word decks, scenarios, topics
"""

import logging
import time

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes.avatar import router as avatar_router
from app.api.routes.curriculum import router as curriculum_router
from app.api.routes.feedback import router as feedback_router
from app.api.routes.sessions import router as sessions_router
from app.api.ws.session import avatar_endpoint, websocket_endpoint
from app.core.config import get_settings
from app.core.database import init_db
from app.core.redis_client import close_redis

settings = get_settings()

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Real-time AI sign language tutor: "
        "hybrid glove + camera recognition → LLM conversation → avatar response"
    ),
    version="0.1.0",
)

# ── CORS ──────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS + ["*"],   # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 웹앱 파일은 매번 최신을 확인하게 (고친 화면이 브라우저 캐시 때문에 안 바뀌어 보이던 문제)
@app.middleware("http")
async def no_cache_app(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/app") and not request.url.path.endswith((".glb", ".wasm", ".tflite", ".data", ".woff2")):
        response.headers["Cache-Control"] = "no-cache"
    return response

# ── REST routers ──────────────────────────────────────────────────
app.include_router(sessions_router, prefix="/api/v1")
app.include_router(feedback_router, prefix="/api/v1")
app.include_router(curriculum_router, prefix="/api/v1")
app.include_router(avatar_router, prefix="/api/v1")

# ── Serve web client static files ────────────────────────────────
import mimetypes
import os

# Windows 레지스트리가 .js 를 text/plain 으로 알려 주면 브라우저가 ES 모듈을 거부한다
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
# 오프라인 묶음(client/vendor, 웹 메타휴먼 GLB)용
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("font/woff2", ".woff2")
mimetypes.add_type("application/octet-stream", ".tflite")
mimetypes.add_type("application/octet-stream", ".data")
mimetypes.add_type("application/octet-stream", ".binarypb")
client_dir = os.path.join(os.path.dirname(__file__), "..", "..", "client")
if os.path.isdir(client_dir):
    app.mount("/app", StaticFiles(directory=client_dir, html=True), name="client")

# ── WebSocket endpoint ────────────────────────────────────────────
@app.websocket("/ws")
async def ws_route(ws: WebSocket):
    await websocket_endpoint(ws)

# Unreal MetaHuman 클라이언트: 튜터 응답(avatar_commands)과 인식 결과를 받는다
@app.websocket("/ws/avatar")
async def ws_avatar_route(ws: WebSocket):
    await avatar_endpoint(ws)

# ── Lifecycle ─────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    logger.info("Starting %s ...", settings.APP_NAME)
    await init_db()
    logger.info("Database initialised")

@app.on_event("shutdown")
async def on_shutdown():
    await close_redis()
    logger.info("Shutdown complete")

# ── Health check ──────────────────────────────────────────────────
_start_time = time.time()

@app.get("/healthz", tags=["meta"])
async def healthz():
    return JSONResponse({
        "status": "ok",
        "uptime_s": round(time.time() - _start_time, 1),
        "app": settings.APP_NAME,
        "debug": settings.DEBUG,
    })
