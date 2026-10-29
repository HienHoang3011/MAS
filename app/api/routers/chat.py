"""Chat API with explicit session ownership and time-scoped memory retrieval."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from openai import APIError
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field
from pymongo.errors import PyMongoError

from app.core.models.chat import ChatMessage, ChatRequest, ChatResponse, ChatSession, LongTermMemory
from app.core.use_cases.chat_system import ChatSystem, GuidancePolicyError
from app.infra.db.agent_sessions import SessionNotFoundError
from app.infra.db.chat_store import SessionConflictError
from app.services.vector_store import KnowledgeIndexNotReadyError

router = APIRouter(prefix="/chat", tags=["chat"])


def get_chat_system(request: Request) -> ChatSystem:
    system: ChatSystem | None = getattr(request.app.state, "chat_system", None)
    if system is None:
        raise HTTPException(
            status_code=503, detail="Chat requires MONGODB_URI and LLM configuration"
        )
    return system


ChatDependency = Annotated[ChatSystem, Depends(get_chat_system)]
UserId = Annotated[str, Query(min_length=1, max_length=128)]


class NewSessionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    user_id: str | None = Field(default=None, min_length=1, max_length=128)


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, system: ChatDependency) -> ChatResponse:
    try:
        return await system.chat(request)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found for this user") from exc
    except SessionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except KnowledgeIndexNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail="MongoDB operation failed") from exc
    except (APIError, ValueError, GuidancePolicyError) as exc:
        raise HTTPException(
            status_code=502, detail="Agent could not produce a valid reply"
        ) from exc


@router.post("/sessions", response_model=ChatSession, status_code=201)
async def create_session(request: NewSessionRequest, system: ChatDependency) -> ChatSession:
    try:
        return await system.store.create_session(request.user_id)
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail="MongoDB operation failed") from exc


@router.get("/sessions/{session_id}", response_model=ChatSession)
async def session(session_id: str, user_id: UserId, system: ChatDependency) -> ChatSession:
    try:
        return await system.store.get_session(user_id, session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found for this user") from exc
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail="MongoDB operation failed") from exc


def validate_period(since: AwareDatetime | None, until: AwareDatetime | None) -> None:
    if since is not None and until is not None and since > until:
        raise HTTPException(status_code=422, detail="since must be at or before until")


@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessage])
async def messages(
    session_id: str,
    user_id: UserId,
    system: ChatDependency,
    after_sequence: Annotated[int, Query(ge=0)] = 0,
    since: AwareDatetime | None = None,
    until: AwareDatetime | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
) -> list[ChatMessage]:
    validate_period(since, until)
    try:
        return await system.store.history(user_id, session_id, after_sequence, since, until, limit)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found for this user") from exc
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail="MongoDB operation failed") from exc


@router.get("/sessions/{session_id}/memories", response_model=list[LongTermMemory])
async def memories(
    session_id: str,
    user_id: UserId,
    system: ChatDependency,
    since: AwareDatetime | None = None,
    until: AwareDatetime | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[LongTermMemory]:
    validate_period(since, until)
    try:
        return await system.store.recall(user_id, session_id, since, until, limit)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found for this user") from exc
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail="MongoDB operation failed") from exc
