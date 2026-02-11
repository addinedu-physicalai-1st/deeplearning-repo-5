"""CRUD 함수"""
from datetime import datetime
from typing import Optional
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from . import models


async def get_members(session: AsyncSession) -> list[models.Member]:
    result = await session.execute(
        select(models.Member).options(selectinload(models.Member.pets))
    )
    return list(result.scalars().all())


async def get_pet_detail(session: AsyncSession, pet_id: int) -> Optional[models.Pet]:
    result = await session.execute(
        select(models.Pet)
        .options(selectinload(models.Pet.member))
        .where(models.Pet.id == pet_id)
    )
    pet = result.scalar_one_or_none()
    if not pet:
        return None

    logs_result = await session.execute(
        select(models.EmotionLog)
        .where(models.EmotionLog.pet_id == pet_id)
        .order_by(desc(models.EmotionLog.id))
        .limit(10)
    )
    pet._recent_logs = list(logs_result.scalars().all())
    return pet


async def get_all_pets(session: AsyncSession) -> list[models.Pet]:
    result = await session.execute(
        select(models.Pet).options(selectinload(models.Pet.member))
    )
    return list(result.scalars().all())


async def create_emotion_log(
    session: AsyncSession,
    pet_id: Optional[int],
    timestamp: datetime,
    object_type: str,
    emotion: str,
    emotion_detail: Optional[str],
    emotion_conf: float,
    fps: int,
    order_text: str,
    raw_json: Optional[dict],
) -> models.EmotionLog:
    log = models.EmotionLog(
        pet_id=pet_id,
        timestamp=timestamp,
        object_type=object_type,
        emotion=emotion,
        emotion_detail=emotion_detail,
        emotion_conf=emotion_conf,
        fps=fps,
        order_text=order_text,
        raw_json=raw_json,
    )
    session.add(log)
    await session.commit()
    await session.refresh(log)
    return log


async def create_action_guide(
    session: AsyncSession,
    log_id: Optional[int],
    emotion_trigger: str,
    vlm_description: Optional[str],
    action_text: str,
    control_cmd: Optional[str],
    control_param: Optional[str],
) -> models.ActionGuide:
    guide = models.ActionGuide(
        log_id=log_id,
        emotion_trigger=emotion_trigger,
        vlm_description=vlm_description,
        action_text=action_text,
        control_cmd=control_cmd,
        control_param=control_param,
    )
    session.add(guide)
    await session.commit()
    await session.refresh(guide)
    return guide


async def get_emotion_logs(
    session: AsyncSession, page: int = 1, page_size: int = 20
) -> tuple[list[models.EmotionLog], int]:
    count_result = await session.execute(select(func.count(models.EmotionLog.id)))
    total = count_result.scalar() or 0

    offset = (page - 1) * page_size
    result = await session.execute(
        select(models.EmotionLog)
        .order_by(desc(models.EmotionLog.id))
        .offset(offset)
        .limit(page_size)
    )
    items = list(result.scalars().all())
    return items, total
