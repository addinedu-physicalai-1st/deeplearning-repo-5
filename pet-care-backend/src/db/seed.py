"""시드 데이터 스크립트"""
import asyncio
from .database import engine, async_session, Base
from .models import Member, Pet


SEED_MEMBERS = [
    {"name": "김하준", "email": "hajun@example.com", "phone": "010-1234-5678"},
    {"name": "이수진", "email": "sujin@example.com", "phone": "010-2345-6789"},
    {"name": "박민준", "email": "minjun@example.com", "phone": "010-3456-7890"},
]

SEED_PETS = [
    {"member_idx": 0, "name": "바둑이", "species": "dog", "breed": "진돗개", "age": 3},
    {"member_idx": 0, "name": "나비", "species": "cat", "breed": "코리안숏헤어", "age": 2},
    {"member_idx": 1, "name": "초코", "species": "dog", "breed": "푸들", "age": 5},
    {"member_idx": 1, "name": "루나", "species": "cat", "breed": "러시안블루", "age": 1},
    {"member_idx": 2, "name": "콩이", "species": "dog", "breed": "시바이누", "age": 4},
]


async def run_seed():
    """시드 데이터 삽입"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session() as session:
        members = []
        for m in SEED_MEMBERS:
            member = Member(**m)
            session.add(member)
            members.append(member)
        await session.flush()

        for p in SEED_PETS:
            pet = Pet(
                member_id=members[p["member_idx"]].id,
                name=p["name"],
                species=p["species"],
                breed=p["breed"],
                age=p["age"],
            )
            session.add(pet)

        await session.commit()
        print(f"Seed complete: {len(members)} members, {len(SEED_PETS)} pets")


if __name__ == "__main__":
    asyncio.run(run_seed())
