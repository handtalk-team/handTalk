from fastapi import APIRouter, HTTPException
from typing import List

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.get("/{session_id}/summary")
async def get_session_summary(session_id: str):
    # TODO: query DB for session summary
    raise HTTPException(status_code=404, detail="Session not found")


@router.get("/{session_id}/error-note")
async def get_error_note(session_id: str):
    """Return the auto-generated error note (오답노트) for a completed session."""
    raise HTTPException(status_code=404, detail="Session not found")
