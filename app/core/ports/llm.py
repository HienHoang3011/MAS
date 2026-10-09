"""LLM port used by the domain workflow."""

from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class StructuredLLM(Protocol):
    async def complete(self, system: str, user: str, response_model: type[T]) -> T: ...
