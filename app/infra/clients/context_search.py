"""Search only the context explicitly supplied for this session."""

import re


class ContextSearchTool:
    name = "context_search"
    description = "Find relevant passages in the context supplied by the user. Argument: query."

    async def run(self, arguments: dict[str, str], context: str) -> str:
        query = arguments.get("query", "").strip()
        if not query:
            return "Error: query is required."
        passages = [part.strip() for part in re.split(r"\n\s*\n", context) if part.strip()]
        words = set(re.findall(r"\w+", query.casefold()))
        ranked = sorted(
            passages,
            key=lambda passage: len(words & set(re.findall(r"\w+", passage.casefold()))),
            reverse=True,
        )
        matches = [part for part in ranked if words & set(re.findall(r"\w+", part.casefold()))]
        return "\n\n".join(matches[:3])[:3000] or "No matching passage in supplied context."
