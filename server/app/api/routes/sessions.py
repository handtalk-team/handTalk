from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/sessions", tags=["sessions"])


class SessionInfo(BaseModel):
    session_id: str
    scenario: str
    status: str


@router.get("/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    # TODO: query DB
    raise HTTPException(status_code=404, detail="Session not found")
