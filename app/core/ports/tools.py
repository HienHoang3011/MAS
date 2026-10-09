"""Restricted execution tool port."""

from typing import Protocol


class AgentTool(Protocol):
    name: str
    description: str

    async def run(self, arguments: dict[str, str], context: str) -> str: ...
