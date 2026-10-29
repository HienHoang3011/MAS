"""Agent API survives a new app/store instance and rejects stale updates."""

import asyncio
import os
from collections import deque
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel
from pymongo import AsyncMongoClient

from app.api.routers.agent import get_agent_system
from app.config.config import get_settings
from app.core.models.agent import (
    Classification,
    ExecutionPlan,
    ExecutionResult,
    FinalAnswer,
    SocraticQuestion,
    Verification,
)
from app.core.use_cases.agent_system import AgentSystem
from app.infra.db.agent_sessions import AgentSessionConflictError, MongoAgentSessionStore
from app.main import create_app

pytestmark = pytest.mark.skipif(
    os.environ.get("MAS_LIVE_MONGO") != "1", reason="Set MAS_LIVE_MONGO=1 for real MongoDB tests"
)


class FakeLLM:
    def __init__(self, responses: list[BaseModel]) -> None:
        self.responses = deque(responses)

    async def complete(self, system: str, user: str, response_model: type[BaseModel]) -> BaseModel:
        response = self.responses.popleft()
        assert isinstance(response, response_model)
        return response


def test_agent_session_survives_new_app_and_rejects_stale_write() -> None:
    asyncio.run(run_scenario())


async def run_scenario() -> None:
    settings = get_settings()
    assert settings.mongodb_uri
    database = "mas_agent_test_" + uuid4().hex[:20]
    first_client: AsyncMongoClient[dict[str, object]] = AsyncMongoClient(
        settings.mongodb_uri, serverSelectionTimeoutMS=15000
    )
    second_client: AsyncMongoClient[dict[str, object]] = AsyncMongoClient(
        settings.mongodb_uri, serverSelectionTimeoutMS=15000
    )
    first_store = MongoAgentSessionStore(first_client, database)
    second_store = MongoAgentSessionStore(second_client, database)
    first_system = AgentSystem(
        FakeLLM(
            [
                Classification(
                    request_type="planning",
                    goal="Choose a date",
                    ready_to_execute=False,
                    missing_information=["preferred day"],
                ),
                SocraticQuestion(question="Which day works?", reason="Need preference"),
            ]
        ),
        first_store,
    )
    second_system = AgentSystem(
        FakeLLM(
            [
                Classification(
                    request_type="planning", goal="Choose a date", ready_to_execute=True
                ),
                ExecutionPlan(approach="Use the preference"),
                ExecutionResult(answer="Friday works."),
                Verification(sufficient=True),
                FinalAnswer(answer="Schedule it for Friday."),
            ]
        ),
        second_store,
    )
    first_app = create_app()
    second_app = create_app()
    first_app.dependency_overrides[get_agent_system] = lambda: first_system
    second_app.dependency_overrides[get_agent_system] = lambda: second_system
    try:
        await first_store.initialize()
        await second_store.initialize()
        async with AsyncClient(
            transport=ASGITransport(app=first_app), base_url="http://test"
        ) as api:
            started = await api.post("/v1/agent/sessions", json={"problem": "Pick a meeting date"})
        assert started.status_code == 201
        session_id = started.json()["id"]
        assert started.json()["revision"] == 0
        stale = await first_store.get(session_id)

        async with AsyncClient(
            transport=ASGITransport(app=second_app), base_url="http://test"
        ) as api:
            fetched = await api.get(f"/v1/agent/sessions/{session_id}")
            assert fetched.status_code == 200
            assert fetched.json()["question"]["question"] == "Which day works?"
            replied = await api.post(
                f"/v1/agent/sessions/{session_id}/replies", json={"answer": "Friday"}
            )
            assert replied.status_code == 200
            assert replied.json()["status"] == "completed"
            assert replied.json()["revision"] == 1
            duplicate = await api.post(
                f"/v1/agent/sessions/{session_id}/replies", json={"answer": "Saturday"}
            )
            assert duplicate.status_code == 409

        stale.revision += 1
        with pytest.raises(AgentSessionConflictError):
            await first_store.replace(stale, expected_revision=0)
        assert (await first_store.get(session_id)).result.answer == "Schedule it for Friday."
    finally:
        await first_client.drop_database(database)
        await first_client.close()
        await second_client.close()
