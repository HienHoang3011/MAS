from fastapi import APIRouter

from app.api.routers import agent

router = APIRouter()
router.include_router(agent.router)


@router.get("/")
def v1_root() -> dict[str, str]:
    return {"message": "API v1"}
