"""Exercise the real OpenAI SDK request and JSON parsing against a local transport."""

import asyncio
import json

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from app.core.models.chat import MemorySummary
from app.services import llm_service


def test_compact_schema_preserves_fields_named_like_schema_metadata() -> None:
    class Metadata(BaseModel):
        title: str
        default: int = 1

    schema = json.loads(llm_service._proxy_schema(Metadata))
    assert "title" not in schema
    assert set(schema["properties"]) == {"title", "default"}
    assert schema["properties"]["title"] == {"type": "string"}
    assert schema["properties"]["default"] == {"type": "integer"}
    assert schema["required"] == ["title"]


@pytest.mark.parametrize(
    "content",
    [
        '{"summary":"Đang học đồ thị"}',
        '```json\n{"summary":"Đang học đồ thị"}\n```',
    ],
)
def test_proxy_validates_plain_chat_json(monkeypatch: pytest.MonkeyPatch, content: str) -> None:
    asyncio.run(proxy_scenario(monkeypatch, content))


async def proxy_scenario(
    monkeypatch: pytest.MonkeyPatch,
    content: str,
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert payload["thinking"] is False
        assert payload["chat_template_kwargs"]["enable_thinking"] is False
        assert "response_format" not in payload
        return httpx.Response(
            200,
            json={
                "id": "demo",
                "created": 1,
                "model": "qwen",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
            },
        )

    client = AsyncOpenAI(
        api_key="test-only",
        base_url="http://test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    monkeypatch.setattr(llm_service, "AsyncOpenAI", lambda **kwargs: client)
    llm = llm_service.ProxyStructuredLLM("test-only", "qwen", "http://test/v1", 120)
    try:
        result = await llm.complete("Summarize", "A conversation", MemorySummary)
        assert result.summary == "Đang học đồ thị"
    finally:
        await llm.close()


def test_proxy_rejects_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError):
        asyncio.run(proxy_scenario(monkeypatch, "not-json"))


@pytest.mark.parametrize(
    ("finish_reason", "content", "expected"),
    [
        ("stop", "Cây là đồ thị liên thông không có chu trình.", True),
        ("length", "Cây là đồ thị", False),
        ("stop", "   ", False),
    ],
)
def test_text_generation_preserves_answer_and_rejects_truncation(
    monkeypatch: pytest.MonkeyPatch, finish_reason: str, content: str, expected: bool
) -> None:
    async def scenario() -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            assert "response_format" not in payload
            assert payload["thinking"] is False
            return httpx.Response(
                200,
                json={
                    "id": "demo",
                    "created": 1,
                    "model": "qwen",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": finish_reason,
                            "message": {"role": "assistant", "content": content},
                        }
                    ],
                },
            )

        client = AsyncOpenAI(
            api_key="test-only",
            base_url="http://test/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )
        monkeypatch.setattr(llm_service, "AsyncOpenAI", lambda **kwargs: client)
        llm = llm_service.ProxyStructuredLLM("test-only", "qwen", "http://test/v1", 120)
        try:
            if expected:
                assert await llm.generate("Explain", "Cây là gì?") == content
            else:
                with pytest.raises(ValueError):
                    await llm.generate("Explain", "Cây là gì?")
        finally:
            await llm.close()

    asyncio.run(scenario())
