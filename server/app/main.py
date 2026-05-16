"""
handTalk — FastAPI Application Entry Point
==========================================

Endpoints
---------
GET  /healthz                   — liveness probe
WS   /ws                        — real-time sign language session
GET  /sessions/{id}             — session metadata (REST)
GET  /feedback/{id}/summary     — post-session summary (REST)
GET  /feedback/{id}/error-note  — auto-generated error note (REST)
"""

import logging
import time

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes.feedback import router as feedback_router
from app.api.routes.sessions import router as sessions_router
from app.api.ws.session import websocket_endpoint
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

# ── REST routers ──────────────────────────────────────────────────
app.include_router(sessions_router, prefix="/api/v1")
app.include_router(feedback_router, prefix="/api/v1")

# ── Serve web client static files ────────────────────────────────
import os
client_dir = os.path.join(os.path.dirname(__file__), "..", "..", "client")
if os.path.isdir(client_dir):
    app.mount("/app", StaticFiles(directory=client_dir, html=True), name="client")

# ── WebSocket endpoint ────────────────────────────────────────────
@app.websocket("/ws")
async def ws_route(ws: WebSocket):
    await websocket_endpoint(ws)

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
