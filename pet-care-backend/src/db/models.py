"""ORM 모델 정의"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, BigInteger, String, Float, Text, DateTime,
    Enum, JSON, ForeignKey, Index,
)
from sqlalchemy.orm import relationship
from .database import Base


class Member(Base):
    __tablename__ = "members"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False)
    email = Column(String(100))
    phone = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)

    pets = relationship("Pet", back_populates="member", cascade="all, delete-orphan")


class Pet(Base):
    __tablename__ = "pets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    member_id = Column(Integer, ForeignKey("members.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(50), nullable=False)
    species = Column(Enum("dog", "cat", name="species_enum"), nullable=False)
    breed = Column(String(100))
    age = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    member = relationship("Member", back_populates="pets")
    emotion_logs = relationship("EmotionLog", back_populates="pet")


class EmotionLog(Base):
    __tablename__ = "emotion_logs"
    __table_args__ = (
        Index("idx_timestamp", "timestamp"),
        Index("idx_pet_emotion", "pet_id", "emotion"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    pet_id = Column(Integer, ForeignKey("pets.id", ondelete="SET NULL"), nullable=True)
    timestamp = Column(DateTime, nullable=False)
    object_type = Column(
        Enum("none", "dog", "cat", name="object_type_enum"),
        nullable=False, default="none",
    )
    emotion = Column(
        Enum("none", "positive", "negative", name="emotion_enum"),
        nullable=False, default="none",
    )
    emotion_detail = Column(String(50))
    emotion_conf = Column(Float, default=0.0)
    fps = Column(Integer, default=0)
    order_text = Column(Text, default="none")
    raw_json = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    pet = relationship("Pet", back_populates="emotion_logs")
    action_guide = relationship("ActionGuide", back_populates="emotion_log", uselist=False)


class ActionGuide(Base):
    __tablename__ = "action_guides"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    log_id = Column(BigInteger, ForeignKey("emotion_logs.id", ondelete="SET NULL"), nullable=True)
    emotion_trigger = Column(String(50), nullable=False)
    vlm_description = Column(Text)
    action_text = Column(Text, nullable=False)
    control_cmd = Column(String(50))
    control_param = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

    emotion_log = relationship("EmotionLog", back_populates="action_guide")
