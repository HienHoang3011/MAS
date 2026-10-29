from fastapi import APIRouter

from app.api.routers import agent, chat

router = APIRouter()
router.include_router(agent.router)
router.include_router(chat.router)


@router.get("/")
def v1_root() -> dict[str, str]:
    return {"message": "API v1"}
