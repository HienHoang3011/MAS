"""Agent session stores with optimistic updates across API workers."""

from typing import Protocol

from pymongo import AsyncMongoClient

from app.core.models.agent import AgentSession


class SessionNotFoundError(Exception):
    pass


class AgentSessionConflictError(Exception):
    """The session changed while an agent was preparing its reply."""


class AgentSessionStore(Protocol):
    async def create(self, session: AgentSession) -> None: ...

    async def get(self, session_id: str) -> AgentSession: ...

    async def replace(self, session: AgentSession, expected_revision: int) -> None: ...


class InMemorySessionStore:
    """A deterministic store for unit tests."""

    def __init__(self) -> None:
        self._sessions: dict[str, AgentSession] = {}

    async def create(self, session: AgentSession) -> None:
        self._sessions[session.id] = session.model_copy(deep=True)

    async def get(self, session_id: str) -> AgentSession:
        try:
            return self._sessions[session_id].model_copy(deep=True)
        except KeyError as exc:
            raise SessionNotFoundError(session_id) from exc

    async def replace(self, session: AgentSession, expected_revision: int) -> None:
        previous = await self.get(session.id)
        if previous.revision != expected_revision:
            raise AgentSessionConflictError("Session changed; resend the reply")
        self._sessions[session.id] = session.model_copy(deep=True)


class MongoAgentSessionStore:
    def __init__(self, client: AsyncMongoClient[dict[str, object]], database: str) -> None:
        self.sessions = client[database]["agent_sessions"]

    async def initialize(self) -> None:
        await self.sessions.create_index("id", unique=True)

    async def create(self, session: AgentSession) -> None:
        await self.sessions.insert_one(session.model_dump())

    async def get(self, session_id: str) -> AgentSession:
        doc = await self.sessions.find_one({"id": session_id})
        if doc is None:
            raise SessionNotFoundError(session_id)
        return AgentSession.model_validate(doc)

    async def replace(self, session: AgentSession, expected_revision: int) -> None:
        result = await self.sessions.replace_one(
            {"id": session.id, "revision": expected_revision}, session.model_dump()
        )
        if result.matched_count:
            return
        if await self.sessions.find_one({"id": session.id}, {"_id": 1}) is None:
            raise SessionNotFoundError(session.id)
        raise AgentSessionConflictError("Session changed; resend the reply")
