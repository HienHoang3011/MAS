from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def ok() -> dict[str, str]:
    return {"status": "ok"}
