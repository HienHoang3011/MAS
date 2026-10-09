"""Data exchanged by the conversational agent system."""

from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field


class SessionStatus(StrEnum):
    WAITING_FOR_USER = "waiting_for_user"
    COMPLETED = "completed"


class Turn(BaseModel):
    role: str
    content: str


class Classification(BaseModel):
    request_type: str
    goal: str
    ready_to_execute: bool
    missing_information: list[str] = Field(default_factory=list)


class SocraticQuestion(BaseModel):
    question: str
    reason: str


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, str] = Field(default_factory=dict)


class ExecutionPlan(BaseModel):
    approach: str
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ToolResult(BaseModel):
    name: str
    output: str


class ExecutionResult(BaseModel):
    answer: str
    evidence: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)


class Verification(BaseModel):
    sufficient: bool
    issues: list[str] = Field(default_factory=list)
    follow_up_question: str | None = None


class FinalAnswer(BaseModel):
    answer: str
    evidence: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class AgentSession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    problem: str
    context: str = ""
    status: SessionStatus = SessionStatus.WAITING_FOR_USER
    turns: list[Turn] = Field(default_factory=list)
    classification: Classification | None = None
    question: SocraticQuestion | None = None
    execution: ExecutionResult | None = None
    verification: Verification | None = None
    result: FinalAnswer | None = None
    clarification_count: int = 0
    execution_count: int = 0
