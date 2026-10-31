"""Validated messages, routing decisions and durable conversation memories."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

SHORT_TERM_LIMIT = 10


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    content: str = Field(min_length=1, max_length=12000)
    mode: Literal["auto", "standard", "fast"] = "auto"
    user_id: str | None = Field(default=None, min_length=1, max_length=128)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def require_session_owner(self) -> Self:
        if self.session_id is not None and self.user_id is None:
            raise ValueError("user_id is required when continuing a session")
        return self


class ChatRoute(StrEnum):
    DIRECT = "direct"
    CONVERSATION = "conversation"
    TUTORING = "tutoring"
    KNOWLEDGE = "knowledge"


class RoutingDecision(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    route: Literal[ChatRoute.CONVERSATION, ChatRoute.TUTORING, ChatRoute.KNOWLEDGE]
    use_fast: bool = False
    fast_answer: str | None = Field(default=None, max_length=3000)
    retrieval_query: str | None = Field(default=None, max_length=2000)
    requirements: list[Annotated[str, Field(min_length=1, max_length=1000)]] = Field(
        default_factory=list, max_length=8
    )


class ChatMessage(BaseModel):
    session_id: str
    user_id: str
    sequence: int = Field(ge=1)
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class ChatSession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    message_count: int = 0
    summarized_through: int = 0
    summary_failures: int = 0
    summary_error_type: str | None = None
    summary_retry_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    summary_lease_until: datetime = Field(default_factory=lambda: datetime.now(UTC))
    summary_owner: str | None = None


class MemorySummary(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    summary: str = Field(min_length=1, max_length=6000)


class LongTermMemory(MemorySummary):
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    user_id: str
    from_sequence: int
    through_sequence: int
    period_start: datetime
    period_end: datetime
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class KnowledgePassage(BaseModel):
    id: str
    content: str
    source: str
    score: float


class SourceReference(BaseModel):
    id: str
    source: str
    score: float


class TextbookChunk(BaseModel):
    chunk_id: str
    source: str
    heading_path: list[str] = Field(default_factory=list)
    content: str = Field(min_length=1)


class EmbeddedChunk(TextbookChunk):
    model: str
    content_hash: str
    embedding: list[float] = Field(min_length=1024, max_length=1024)


class RequirementCoverage(BaseModel):
    requirement_index: int = Field(ge=0, description="Zero-based position in requirements.")
    satisfied: bool
    answer_block_ids: list[int] = Field(max_length=128)
    issue: str | None = Field(default=None, max_length=2000)


class ReplyCoverage(BaseModel):
    checks: list[RequirementCoverage] = Field(min_length=1, max_length=8)
    feedback: str | None = Field(default=None, max_length=3000)


class FactReview(BaseModel):
    block_id: int = Field(ge=0)
    claim: str = Field(min_length=1, max_length=500)
    assessment: str = Field(min_length=1, max_length=500)
    correct: bool
    issue: str | None = Field(default=None, max_length=1000)


class ReplyReview(ReplyCoverage):
    facts: list[FactReview] = Field(min_length=1, max_length=128)


class FactIssue(BaseModel):
    block_id: int = Field(ge=0)
    issue: str = Field(min_length=1, max_length=500)


class SourceCitation(BaseModel):
    # Invalid/empty evidence is reported by review_issues so the draft can be rewritten.
    answer_block_ids: list[int] = Field(max_length=128)
    source_index: int = Field(ge=0)
    source_block_ids: list[int] = Field(max_length=128)


class GuidanceReview(BaseModel):
    citations: list[SourceCitation] = Field(max_length=128)
    grounded: bool
    pedagogically_safe: bool
    feedback: str | None = Field(default=None, max_length=3000)


class KnowledgeReview(ReplyCoverage, GuidanceReview):
    """One audit of coverage, factual accuracy, sources and tutoring boundaries."""

    correct_block_ids: list[int] = Field(max_length=128)
    incorrect_blocks: list[FactIssue] = Field(max_length=128)


class TutoringReview(ReplyCoverage, GuidanceReview):
    """One audit of useful guidance, sources and tutoring boundaries."""


class ChatResponse(BaseModel):
    user_id: str
    session_id: str
    route: ChatRoute
    answer: str
    sources: list[SourceReference]
    summary_pending: bool
