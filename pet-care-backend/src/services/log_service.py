"""로그 포맷팅 + DB 저장 로직 (v3 신규 작성)"""
import logging
from datetime import datetime
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from src.db import crud

logger = logging.getLogger(__name__)


async def save_emotion_log(
    session: AsyncSession,
    log_data: dict,
    pet_id: Optional[int] = None,
) -> Optional[int]:
    """5초 주기 로그를 DB에 저장하고 log_id 반환"""
    try:
        timestamp_str = log_data.get("timestamp", "")
        try:
            ts = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            ts = datetime.utcnow()

        log = await crud.create_emotion_log(
            session=session,
            pet_id=pet_id,
            timestamp=ts,
            object_type=log_data.get("object", "none"),
            emotion=log_data.get("emotion", "none"),
            emotion_detail=log_data.get("emotion_detail"),
            emotion_conf=log_data.get("emotion_conf", 0.0),
            fps=log_data.get("fps", 0),
            order_text=log_data.get("order", "none"),
            raw_json=log_data,
        )
        return log.id
    except Exception as e:
        logger.error(f"EmotionLog 저장 실패: {e}")
        return None


async def save_action_guide(
    session: AsyncSession,
    log_id: Optional[int],
    guide_data: dict,
) -> None:
    """GenAI 행동지시를 DB에 저장"""
    try:
        control = guide_data.get("control") or {}
        await crud.create_action_guide(
            session=session,
            log_id=log_id,
            emotion_trigger=guide_data.get("emotion_trigger", "unknown"),
            vlm_description=guide_data.get("vlm_description"),
            action_text=guide_data.get("action", ""),
            control_cmd=control.get("cmd") if isinstance(control, dict) else None,
            control_param=control.get("param") if isinstance(control, dict) else None,
        )
    except Exception as e:
        logger.error(f"ActionGuide 저장 실패: {e}")
