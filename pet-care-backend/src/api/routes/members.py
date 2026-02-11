"""GET /api/members, GET /api/pets/{id} - 회원/반려동물 조회"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from src.db.database import get_session
from src.db import crud, schemas

router = APIRouter()


@router.get("/members", response_model=list[schemas.MemberOut])
async def list_members(session: AsyncSession = Depends(get_session)):
    return await crud.get_members(session)


@router.get("/pets", response_model=list[schemas.PetOut])
async def list_pets(session: AsyncSession = Depends(get_session)):
    return await crud.get_all_pets(session)


@router.get("/pets/{pet_id}")
async def get_pet(pet_id: int, session: AsyncSession = Depends(get_session)):
    pet = await crud.get_pet_detail(session, pet_id)
    if not pet:
        raise HTTPException(status_code=404, detail="Pet not found")

    recent_logs = [schemas.EmotionLogOut.model_validate(l) for l in getattr(pet, "_recent_logs", [])]
    return {
        "id": pet.id,
        "member_id": pet.member_id,
        "name": pet.name,
        "species": pet.species,
        "breed": pet.breed,
        "age": pet.age,
        "created_at": pet.created_at,
        "member_name": pet.member.name if pet.member else "",
        "recent_logs": recent_logs,
    }
