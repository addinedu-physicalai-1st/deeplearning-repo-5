"""Pydantic 스키마"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class MemberOut(BaseModel):
    id: int
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class PetOut(BaseModel):
    id: int
    member_id: int
    name: str
    species: str
    breed: Optional[str] = None
    age: Optional[int] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class PetDetailOut(PetOut):
    member_name: str
    recent_logs: list["EmotionLogOut"] = []


class EmotionLogOut(BaseModel):
    id: int
    pet_id: Optional[int] = None
    timestamp: datetime
    object_type: str
    emotion: str
    emotion_detail: Optional[str] = None
    emotion_conf: float
    fps: int
    order_text: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ActionGuideOut(BaseModel):
    id: int
    log_id: Optional[int] = None
    emotion_trigger: str
    vlm_description: Optional[str] = None
    action_text: str
    control_cmd: Optional[str] = None
    control_param: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class LogPageOut(BaseModel):
    items: list[EmotionLogOut]
    total: int
    page: int
    page_size: int
