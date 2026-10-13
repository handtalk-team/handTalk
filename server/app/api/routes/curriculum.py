from fastapi import APIRouter

from app.services.tutor.curriculum import catalog

router = APIRouter(prefix="/curriculum", tags=["curriculum"])


@router.get("")
async def get_curriculum():
    """학습 모드, 단어 묶음, 상황 시나리오, 프리토킹 주제 (웹 앱 화면 구성용)"""
    return catalog()
