"""HTTP interface for the conversational agent system."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from pymongo.errors import PyMongoError

from app.core.models.agent import AgentSession
from app.core.use_cases.agent_system import AgentSystem, InvalidSessionStateError
from app.infra.db.agent_sessions import AgentSessionConflictError, SessionNotFoundError

router = APIRouter(prefix="/agent", tags=["agent"])


class StartRequest(BaseModel):
    problem: str = Field(min_length=1, max_length=12000)
    context: str = Field(default="", max_length=12000)


class ReplyRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=6000)


async def get_agent_system(request: Request) -> AgentSystem:
    system: AgentSystem | None = getattr(request.app.state, "agent_system", None)
    if system is None:
        raise HTTPException(status_code=503, detail="Agent requires MongoDB and LLM configuration")
    return system


AgentSystemDependency = Annotated[AgentSystem, Depends(get_agent_system)]


@router.post("/sessions", response_model=AgentSession, status_code=status.HTTP_201_CREATED)
async def start_session(request: StartRequest, system: AgentSystemDependency) -> AgentSession:
    try:
        return await system.start(request.problem, request.context)
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail="MongoDB operation failed") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Agent processing failed") from exc


@router.get("/sessions/{session_id}", response_model=AgentSession)
async def get_session(session_id: str, system: AgentSystemDependency) -> AgentSession:
    try:
        return await system.get(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail="MongoDB operation failed") from exc


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
    except AgentSessionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail="MongoDB operation failed") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Agent processing failed") from exc
