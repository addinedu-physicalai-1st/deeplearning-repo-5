"""GET /api/logs - DB 로그 이력 조회 (페이지네이션)"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from src.db.database import get_session
from src.db import crud, schemas

router = APIRouter()


@router.get("/logs", response_model=schemas.LogPageOut)
async def list_logs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_session),
):
    items, total = await crud.get_emotion_logs(session, page, page_size)
    return schemas.LogPageOut(
        items=[schemas.EmotionLogOut.model_validate(i) for i in items],
        total=total,
        page=page,
        page_size=page_size,
    )
