"""Process-local session storage and per-session serialization."""

import asyncio

from app.core.models.agent import AgentSession


class SessionNotFoundError(Exception):
    pass


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, AgentSession] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def create(self, problem: str, context: str) -> AgentSession:
        session = AgentSession(problem=problem, context=context)
        self._sessions[session.id] = session
        self._locks[session.id] = asyncio.Lock()
        return session

    def get(self, session_id: str) -> AgentSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise SessionNotFoundError(session_id) from exc

    def lock(self, session_id: str) -> asyncio.Lock:
        self.get(session_id)
        return self._locks[session_id]

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._locks.pop(session_id, None)

    def replace(self, session: AgentSession) -> None:
        self.get(session.id)
        self._sessions[session.id] = session
