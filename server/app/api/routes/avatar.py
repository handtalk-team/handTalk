"""아바타 직접 조작 (시연/점검용): 글로스 목록을 보내면 실시간 선생님이 바로 수어로 한다."""
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from app.api.ws.manager import manager
from app.models.schemas.ws_messages import LLMResponse
from app.services.avatar.controller import AvatarController
from app.services.tutor.curriculum import can_sign

router = APIRouter(prefix="/avatar", tags=["avatar"])


class SayRequest(BaseModel):
    gloss: List[str]
    text: Optional[str] = None


@router.post("/say")
async def say(req: SayRequest):
    known = [g for g in req.gloss if can_sign(g)]
    msg = LLMResponse(text=req.text or " ".join(req.gloss), gloss=known, kind="reply",
                      avatar_commands=AvatarController.from_glosses(known))
    await manager.broadcast_avatar(msg.model_dump())
    return {"played": known, "unknown": [g for g in req.gloss if g not in known]}
