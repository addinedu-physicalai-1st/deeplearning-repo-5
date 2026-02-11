"""GET /api/status - 현재 파이프라인 상태 JSON"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/status")
async def get_status():
    from src.api.main import get_app_state
    state = get_app_state()

    if not state.ready or not state.streamer:
        return {
            "fps": 0,
            "detected": False,
            "object_type": "none",
            "emotion": "초기화 중...",
            "emotion_detail": None,
            "emotion_conf": 0.0,
            "order": "none",
            "member_name": "",
            "pet_name": "",
            "breed": "",
        }
    return state.streamer.get_status()
