"""OpenAI adapter for validated JSON responses."""

from openai import AsyncOpenAI

from app.core.ports.llm import T


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
