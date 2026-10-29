"""MongoDB persistence with atomic turns and durable, leased summary jobs."""

from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import uuid4

from pymongo import AsyncMongoClient, ReturnDocument
from pymongo.asynchronous.client_session import AsyncClientSession

from app.core.models.chat import (
    SHORT_TERM_LIMIT,
    ChatMessage,
    ChatSession,
    LongTermMemory,
)
from app.infra.db.agent_sessions import SessionNotFoundError


class SessionConflictError(Exception):
    """Another request committed a turn based on the same conversation revision."""


class MongoChatStore:
    def __init__(self, client: AsyncMongoClient[dict[str, object]], database: str) -> None:
        self.client = client
        db = client[database]
        self.sessions = db["chat_sessions"]
        self.messages = db["chat_messages"]
        self.memories = db["chat_memories"]

    async def initialize(self) -> None:
        await self.client.admin.command("ping")
        await self.sessions.create_index("id", unique=True)
        await self.sessions.create_index([("user_id", 1), ("updated_at", -1)])
        await self.messages.create_index([("session_id", 1), ("sequence", 1)], unique=True)
        await self.messages.create_index([("user_id", 1), ("session_id", 1), ("created_at", 1)])
        await self.memories.create_index([("session_id", 1), ("through_sequence", 1)], unique=True)
        await self.memories.create_index([("user_id", 1), ("session_id", 1), ("period_end", -1)])

    async def get_session(self, user_id: str, session_id: str) -> ChatSession:
        doc = await self.sessions.find_one({"id": session_id, "user_id": user_id})
        if doc is None:
            raise SessionNotFoundError(session_id)
        return ChatSession.model_validate(doc)

    async def resolve_session(self, user_id: str | None, session_id: str | None) -> ChatSession:
        if session_id is not None:
            if user_id is None:
                raise ValueError("user_id is required to continue a session")
            return await self.get_session(user_id, session_id)
        if user_id is not None:
            doc = await self.sessions.find_one(
                {"user_id": user_id}, sort=[("updated_at", -1), ("id", -1)]
            )
            if doc is not None:
                return ChatSession.model_validate(doc)
        return await self.create_session(user_id)

    async def create_session(self, user_id: str | None) -> ChatSession:
        session = ChatSession(user_id=user_id or "guest_" + uuid4().hex)
        await self.sessions.insert_one(session.model_dump())
        return session

    async def recent_messages(self, session: ChatSession) -> list[ChatMessage]:
        docs = (
            await self.messages.find(
                {
                    "session_id": session.id,
                    "user_id": session.user_id,
                    "sequence": {"$lte": session.message_count},
                }
            )
            .sort("sequence", -1)
            .limit(SHORT_TERM_LIMIT)
            .to_list()
        )
        return [ChatMessage.model_validate(doc) for doc in reversed(docs)]

    async def message_range(
        self, session: ChatSession, after: int, through: int
    ) -> list[ChatMessage]:
        docs = (
            await self.messages.find(
                {
                    "session_id": session.id,
                    "user_id": session.user_id,
                    "sequence": {"$gt": after, "$lte": through},
                }
            )
            .sort("sequence", 1)
            .limit(100)
            .to_list()
        )
        return [ChatMessage.model_validate(doc) for doc in docs]

    async def history(
        self,
        user_id: str,
        session_id: str,
        after_sequence: int,
        since: datetime | None,
        until: datetime | None,
        limit: int,
    ) -> list[ChatMessage]:
        await self.get_session(user_id, session_id)
        query: dict[str, object] = {
            "user_id": user_id,
            "session_id": session_id,
            "sequence": {"$gt": after_sequence},
        }
        times: dict[str, datetime] = {}
        if since is not None:
            times["$gte"] = since
        if until is not None:
            times["$lte"] = until
        if times:
            query["created_at"] = times
        docs = await self.messages.find(query).sort("sequence", 1).limit(limit).to_list()
        return [ChatMessage.model_validate(doc) for doc in docs]

    async def commit_turn(self, session: ChatSession, content: str, answer: str) -> None:
        now = datetime.now(UTC)
        entries: list[tuple[int, Literal["user", "assistant"], str]] = [
            (1, "user", content),
            (2, "assistant", answer),
        ]
        messages = [
            ChatMessage(
                session_id=session.id,
                user_id=session.user_id,
                sequence=session.message_count + offset,
                role=role,
                content=text,
                created_at=now,
            ).model_dump()
            for offset, role, text in entries
        ]

        async def commit(transaction: AsyncClientSession) -> None:
            result = await self.sessions.update_one(
                {
                    "id": session.id,
                    "user_id": session.user_id,
                    "message_count": session.message_count,
                },
                {"$inc": {"message_count": 2}, "$set": {"updated_at": now}},
                session=transaction,
            )
            if result.modified_count != 1:
                raise SessionConflictError("Session changed; resend the message with fresh context")
            await self.messages.insert_many(messages, session=transaction)

        # Retry only transaction errors that the MongoDB driver labels transient.
        async with self.client.start_session() as transaction:
            await transaction.with_transaction(commit)

    async def recall(
        self,
        user_id: str,
        session_id: str,
        since: datetime | None,
        until: datetime | None,
        limit: int,
    ) -> list[LongTermMemory]:
        await self.get_session(user_id, session_id)
        query: dict[str, object] = {"user_id": user_id, "session_id": session_id}
        if since is not None:
            query["period_end"] = {"$gte": since}
        if until is not None:
            query["period_start"] = {"$lte": until}
        docs = await self.memories.find(query).sort("through_sequence", -1).limit(limit).to_list()
        return [LongTermMemory.model_validate(doc) for doc in reversed(docs)]

    async def claim_summary(self) -> ChatSession | None:
        now = datetime.now(UTC)
        doc = await self.sessions.find_one_and_update(
            {
                "$expr": {
                    "$gt": [
                        {"$subtract": ["$message_count", "$summarized_through"]},
                        SHORT_TERM_LIMIT,
                    ]
                },
                "summary_failures": {"$lt": 3},
                "summary_retry_at": {"$lte": now},
                "summary_lease_until": {"$lte": now},
            },
            {
                "$set": {
                    "summary_owner": uuid4().hex,
                    "summary_lease_until": now + timedelta(minutes=16),
                }
            },
            sort=[("updated_at", 1)],
            return_document=ReturnDocument.AFTER,
        )
        return ChatSession.model_validate(doc) if doc else None

    async def save_summary(self, session: ChatSession, memory: LongTermMemory) -> None:
        async def commit(transaction: AsyncClientSession) -> None:
            result = await self.sessions.update_one(
                {
                    "id": session.id,
                    "summary_owner": session.summary_owner,
                    "summarized_through": session.summarized_through,
                },
                {
                    "$set": {
                        "summarized_through": memory.through_sequence,
                        "summary_failures": 0,
                        "summary_error_type": None,
                        "summary_owner": None,
                        "summary_lease_until": datetime.now(UTC),
                    }
                },
                session=transaction,
            )
            if result.modified_count != 1:
                raise SessionConflictError("Summary lease or sequence changed")
            await self.memories.insert_one(memory.model_dump(), session=transaction)

        async with self.client.start_session() as transaction:
            await transaction.with_transaction(commit)

    async def fail_summary(
        self,
        session: ChatSession,
        error_type: str,
        retry_at: datetime | None,
    ) -> None:
        await self.sessions.update_one(
            {"id": session.id, "summary_owner": session.summary_owner},
            {
                "$set": {
                    "summary_failures": session.summary_failures + 1 if retry_at else 3,
                    "summary_error_type": error_type,
                    "summary_retry_at": retry_at or datetime.now(UTC),
                    "summary_lease_until": datetime.now(UTC),
                    "summary_owner": None,
                }
            },
        )

    async def release_summary(self, session: ChatSession) -> None:
        await self.sessions.update_one(
            {"id": session.id, "summary_owner": session.summary_owner},
            {"$set": {"summary_owner": None, "summary_lease_until": datetime.now(UTC)}},
        )
