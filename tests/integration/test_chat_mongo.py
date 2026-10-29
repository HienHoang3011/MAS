"""Opt-in integration against MongoDB, using a new disposable database and fake LLM."""

import asyncio
import os
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from pymongo import AsyncMongoClient

from app.api.routers.chat import get_chat_system
from app.config.config import get_settings
from app.core.models.chat import (
    ChatRoute,
    MemorySummary,
    RoutingDecision,
)
from app.core.use_cases.chat_system import ChatSystem
from app.infra.db.chat_store import MongoChatStore, SessionConflictError
from app.main import create_app
from app.services.embedding_service import BgeM3Embedder
from app.services.vector_store import BgeKnowledgeRetriever, index_textbook
from tests.unit.test_chat_system import (
    ScriptedLLM,
    approved_review,
    dependencies,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("MAS_LIVE_MONGO") != "1", reason="Set MAS_LIVE_MONGO=1 for real MongoDB tests"
)


def test_real_mongo_chat_memory_and_session_isolation() -> None:
    asyncio.run(run_scenario())


async def run_scenario() -> None:
    settings = get_settings()
    assert settings.mongodb_uri
    mongo: AsyncMongoClient[dict[str, object]] = AsyncMongoClient(
        settings.mongodb_uri, tz_aware=True, serverSelectionTimeoutMS=15000
    )
    database = "mas_test_" + uuid4().hex[:24]
    store = MongoChatStore(mongo, database)
    _, retriever = dependencies()
    llm = ScriptedLLM(
        [
            response
            for i in range(7)
            for response in (
                RoutingDecision(route=ChatRoute.CONVERSATION, requirements=["Respond naturally"]),
                f"Chào bạn, lượt {i}",
                approved_review(f"Chào bạn, lượt {i}"),
            )
        ]
        + [MemorySummary(summary="Người học muốn tìm hiểu lý thuyết đồ thị.")]
    )
    system = ChatSystem(llm, store, retriever)
    app = create_app()
    app.dependency_overrides[get_chat_system] = lambda: system
    try:
        await store.initialize()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            first = await client.post("/v1/chat", json={"content": "Chào bạn"})
            assert first.status_code == 200
            identity = {k: first.json()[k] for k in ("user_id", "session_id")}
            for _ in range(5):
                response = await client.post("/v1/chat", json={**identity, "content": "Tiếp tục"})
                assert response.status_code == 200
            assert response.json()["summary_pending"]
            session = await store.get_session(**identity)
            recent = await store.recent_messages(session)
            assert [m.sequence for m in recent] == list(range(3, 13))
            # A known user without a session_id resumes their latest session.
            resumed = await client.post(
                "/v1/chat", json={"content": "Nhớ mình chứ?", "user_id": identity["user_id"]}
            )
            assert resumed.json()["session_id"] == identity["session_id"]
            assert await system.summarize_next()
            memories = await store.recall(**identity, since=None, until=None, limit=10)
            assert len(memories) == 1
            assert memories[0].through_sequence == 4
            # New requests include persisted long-term memory and only ten recent messages.
            now = datetime.now(UTC)
            filtered = await client.get(
                f"/v1/chat/sessions/{identity['session_id']}/memories",
                params={
                    "user_id": identity["user_id"],
                    "since": (now + timedelta(days=1)).isoformat(),
                },
            )
            assert filtered.status_code == 200 and filtered.json() == []
            history = await client.get(
                f"/v1/chat/sessions/{identity['session_id']}/messages",
                params={"user_id": identity["user_id"], "after_sequence": 10},
            )
            assert [m["sequence"] for m in history.json()] == [11, 12, 13, 14]
            denied = await client.get(
                f"/v1/chat/sessions/{identity['session_id']}/messages",
                params={"user_id": "another-user"},
            )
            assert denied.status_code == 404
            other = await client.post("/v1/chat/sessions", json={})
            assert other.status_code == 201 and other.json()["id"] != identity["session_id"]
            # A stale revision cannot append or overwrite a committed turn.
            with pytest.raises(SessionConflictError):
                await store.commit_turn(session, "stale", "stale")
            refreshed = await store.get_session(**identity)
            assert refreshed.message_count == 14
            assert refreshed.summarized_through == 4
            assert await store.messages.count_documents({}) == 14
        source = mongo[database]["textbook"]
        await source.insert_many(
            [
                {
                    "chunk_id": "handshake",
                    "content": "Tổng bậc bằng hai lần số cạnh.",
                    "source": "test-textbook",
                    "heading_path": ["Bậc của đỉnh"],
                },
                {
                    "chunk_id": "path",
                    "content": "Đường đi trong đồ thị.",
                    "source": "test-textbook",
                    "heading_path": ["Đường đi"],
                },
            ]
        )
        embedder = Mock(spec=BgeM3Embedder)
        embedder.model_name = "BAAI/bge-m3"
        embedder.encode = AsyncMock(
            side_effect=lambda texts: [[1.0, 0.0] + [0.0] * 1022 for text in texts]
        )
        assert await index_textbook(mongo, database, "textbook", database, embedder) == 2
        assert await index_textbook(mongo, database, "textbook", database, embedder) == 0
        assert embedder.encode.await_count == 1
        await source.update_one({"chunk_id": "path"}, {"$set": {"content": "Nội dung mới."}})
        assert await index_textbook(mongo, database, "textbook", database, embedder) == 1
        retrieved = await BgeKnowledgeRetriever(mongo, database, embedder).search("đường đi")
        assert len(retrieved) == 2
        assert "Nội dung mới." in [passage.content for passage in retrieved]
        assert await source.count_documents({"embedding": {"$exists": True}}) == 0
    finally:
        await mongo.drop_database(database)
        await mongo.close()
