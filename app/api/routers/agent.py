"""HTTP interface for the conversational agent system."""

from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.config.config import get_settings
from app.core.models.agent import AgentSession
from app.core.use_cases.agent_system import AgentSystem, InvalidSessionStateError
from app.infra.clients.context_search import ContextSearchTool
from app.infra.db.agent_sessions import InMemorySessionStore, SessionNotFoundError
from app.services.llm_service import OpenAIStructuredLLM

router = APIRouter(prefix="/agent", tags=["agent"])


class StartRequest(BaseModel):
    problem: str = Field(min_length=1, max_length=12000)
    context: str = Field(default="", max_length=12000)


class ReplyRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=6000)


@lru_cache(maxsize=1)
def _build_agent_system() -> AgentSystem:
    settings = get_settings()
    if settings.llm_provider == "gemini":
        api_key = settings.gemini_api_key
        model = settings.gemini_model
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        key_name = "GEMINI_API_KEY"
    else:
        api_key = settings.openai_api_key
        model = settings.openai_model
        base_url = None
        key_name = "OPENAI_API_KEY"
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{key_name} is not configured",
        )
    return AgentSystem(
        llm=OpenAIStructuredLLM(api_key, model, base_url=base_url),
        store=InMemorySessionStore(),
        tools=[ContextSearchTool()],
        max_clarifications=settings.agent_max_clarifications,
        max_executions=settings.agent_max_executions,
    )


async def get_agent_system() -> AgentSystem:
    return _build_agent_system()


AgentSystemDependency = Annotated[AgentSystem, Depends(get_agent_system)]


@router.post("/sessions", response_model=AgentSession, status_code=status.HTTP_201_CREATED)
async def start_session(request: StartRequest, system: AgentSystemDependency) -> AgentSession:
    try:
        return await system.start(request.problem, request.context)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Agent processing failed") from exc


@router.get("/sessions/{session_id}", response_model=AgentSession)
async def get_session(session_id: str, system: AgentSystemDependency) -> AgentSession:
    try:
        return system.get(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc


@router.post("/sessions/{session_id}/replies", response_model=AgentSession)
async def reply(
    session_id: str, request: ReplyRequest, system: AgentSystemDependency
) -> AgentSession:
    try:
        return await system.reply(session_id, request.answer)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    except InvalidSessionStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Agent processing failed") from exc
