"""HTTP round trip through the agent router with a deterministic LLM."""

import asyncio
from collections import deque

from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel

from app.api.routers.agent import get_agent_system
from app.core.models.agent import (
    Classification,
    ExecutionPlan,
    ExecutionResult,
    FinalAnswer,
    SocraticQuestion,
    Verification,
)
from app.core.use_cases.agent_system import AgentSystem
from app.infra.db.agent_sessions import InMemorySessionStore
from app.main import create_app


class FakeLLM:
    def __init__(self, responses: list[BaseModel]) -> None:
        self.responses = deque(responses)

    async def complete(self, system: str, user: str, response_model: type[BaseModel]) -> BaseModel:
        response = self.responses.popleft()
        assert isinstance(response, response_model)
        return response


def test_agent_session_http_round_trip() -> None:
    asyncio.run(_agent_session_http_round_trip())


async def _agent_session_http_round_trip() -> None:
    llm = FakeLLM(
        [
            Classification(
                request_type="planning",
                goal="Choose a date",
                ready_to_execute=False,
                missing_information=["preferred day"],
            ),
            SocraticQuestion(question="Which day works?", reason="Need preference"),
            Classification(request_type="planning", goal="Choose a date", ready_to_execute=True),
            ExecutionPlan(approach="Use the preference"),
            ExecutionResult(answer="Friday works."),
            Verification(sufficient=True),
            FinalAnswer(answer="Schedule it for Friday."),
        ]
    )
    app = create_app()
    system = AgentSystem(llm, InMemorySessionStore())

    async def fake_agent_system() -> AgentSystem:
        return system

    app.dependency_overrides[get_agent_system] = fake_agent_system

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        started = await client.post(
            "/v1/agent/sessions", json={"problem": "Pick a meeting date"}
        )
        assert started.status_code == 201
        session_id = started.json()["id"]
        assert started.json()["question"]["question"] == "Which day works?"

        replied = await client.post(
            f"/v1/agent/sessions/{session_id}/replies", json={"answer": "Friday"}
        )
        assert replied.status_code == 200
        assert replied.json()["status"] == "completed"
        assert replied.json()["result"]["answer"] == "Schedule it for Friday."

        fetched = await client.get(f"/v1/agent/sessions/{session_id}")
        assert fetched.status_code == 200
        assert fetched.json()["result"] == replied.json()["result"]

        duplicate_reply = await client.post(
            f"/v1/agent/sessions/{session_id}/replies", json={"answer": "Saturday"}
        )
        assert duplicate_reply.status_code == 409

        unknown = await client.get("/v1/agent/sessions/unknown")
        assert unknown.status_code == 404
