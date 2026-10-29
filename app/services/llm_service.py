"""Adapters for validated responses from OpenAI-compatible providers."""

import json

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from app.config.config import Settings
from app.core.ports.llm import T


def _completion_text(response: ChatCompletion) -> str:
    if not response.choices:
        raise ValueError("The model returned no choices")
    choice = response.choices[0]
    if choice.finish_reason != "stop":
        raise ValueError(f"The model returned an incomplete response ({choice.finish_reason})")
    if not choice.message.content or not choice.message.content.strip():
        raise ValueError("The model returned empty content")
    return choice.message.content.strip()


class OpenAIStructuredLLM:
    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    async def complete(self, system: str, user: str, response_model: type[T]) -> T:
        response = await self._client.beta.chat.completions.parse(
            model=self._model,
            response_format=response_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("The model returned no structured response")
        return parsed

    async def generate(self, system: str, user: str) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        return _completion_text(response)

    async def close(self) -> None:
        await self._client.close()


class ProxyStructuredLLM:
    """Generate text normally and validate control responses locally against their schema."""

    def __init__(self, api_key: str, model: str, base_url: str, timeout: float) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=2
        )
        self._model = model

    async def complete(self, system: str, user: str, response_model: type[T]) -> T:
        schema = response_model.model_json_schema()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": system
                    + "\nReturn exactly one compact JSON object matching this schema. "
                    "No Markdown, commentary, or trailing text. Stop after the closing brace: "
                    + json.dumps(schema, ensure_ascii=False, separators=(",", ":")),
                },
                {"role": "user", "content": user},
            ],
            # Native guided JSON decoding on this proxy's Qwen build produced whitespace loops.
            temperature=0.2,
            max_tokens=4096,
            extra_body={
                "thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        content = _completion_text(response)
        # Qwen may wrap JSON in a Markdown fence even when thinking is disabled.
        content = content.strip()
        if content.startswith("```json\n") and content.endswith("```"):
            content = content[8:-3].strip()
        elif content.startswith("```\n") and content.endswith("```"):
            content = content[4:-3].strip()
        return response_model.model_validate_json(content)

    async def generate(self, system: str, user: str) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=2048,
            extra_body={
                "thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        return _completion_text(response)

    async def close(self) -> None:
        await self._client.close()


def build_llm(settings: Settings) -> OpenAIStructuredLLM | ProxyStructuredLLM:
    if settings.llm_provider == "proxy":
        if not settings.llm_api_key or not settings.llm_base_url:
            raise ValueError("LLM_API_KEY and LLM_BASE_URL are required for the proxy")
        return ProxyStructuredLLM(
            settings.llm_api_key, settings.llm_model, settings.llm_base_url, settings.llm_timeout
        )
    if settings.llm_provider == "gemini":
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not configured")
        return OpenAIStructuredLLM(
            settings.gemini_api_key,
            settings.gemini_model,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not configured")
    return OpenAIStructuredLLM(settings.openai_api_key, settings.openai_model)
