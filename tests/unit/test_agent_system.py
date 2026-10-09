"""Behavior tests for the bounded agent workflow without network calls."""

from collections import deque
from unittest import IsolatedAsyncioTestCase

from pydantic import BaseModel

from app.core.models.agent import (
    Classification,
    ExecutionPlan,
    ExecutionResult,
    FinalAnswer,
    SessionStatus,
    SocraticQuestion,
    ToolCall,
    Verification,
)
from app.core.use_cases.agent_system import AgentSystem, InvalidSessionStateError
from app.infra.clients.context_search import ContextSearchTool
from app.infra.db.agent_sessions import InMemorySessionStore


class FakeLLM:
    def __init__(self, responses: list[BaseModel]) -> None:
        self.responses = deque(responses)
        self.prompts: list[str] = []

    async def complete(self, system: str, user: str, response_model: type[BaseModel]) -> BaseModel:
        self.prompts.append(user)
        response = self.responses.popleft()
        assert isinstance(response, response_model)
        return response


class AgentSystemTests(IsolatedAsyncioTestCase):
    async def test_ready_request_executes_tool_verifies_and_finishes(self) -> None:
        llm = FakeLLM(
            [
                Classification(
                    request_type="question", goal="Find deadline", ready_to_execute=True
                ),
                ExecutionPlan(
                    approach="Search the supplied note",
                    tool_calls=[ToolCall(name="context_search", arguments={"query": "deadline"})],
                ),
                ExecutionResult(answer="Friday", evidence=["Deadline is Friday."]),
                Verification(sufficient=True),
                FinalAnswer(answer="The deadline is Friday.", evidence=["Deadline is Friday."]),
            ]
        )
        system = AgentSystem(llm, InMemorySessionStore(), [ContextSearchTool()])

        session = await system.start("When is the deadline?", "Deadline is Friday.")

        self.assertEqual(session.status, SessionStatus.COMPLETED)
        self.assertEqual(session.execution_count, 1)
        self.assertEqual(session.execution.tool_results[0].output, "Deadline is Friday.")
        self.assertEqual(session.result.answer, "The deadline is Friday.")
        self.assertIn("Deadline is Friday.", llm.prompts[2])
        with self.assertRaises(InvalidSessionStateError):
            await system.reply(session.id, "another answer")

    async def test_clarification_then_critic_retry(self) -> None:
        llm = FakeLLM(
            [
                Classification(
                    request_type="decision",
                    goal="Choose an option",
                    ready_to_execute=False,
                    missing_information=["budget"],
                ),
                SocraticQuestion(question="What is your budget?", reason="Needed for choice"),
                Classification(request_type="decision", goal="Choose", ready_to_execute=True),
                ExecutionPlan(approach="Compare options"),
                ExecutionResult(answer="Option A"),
                Verification(sufficient=False, issues=["No reason given"]),
                ExecutionPlan(approach="Explain the comparison"),
                ExecutionResult(answer="Option A fits the budget"),
                Verification(sufficient=True),
                FinalAnswer(answer="Choose A because it fits your budget."),
            ]
        )
        system = AgentSystem(llm, InMemorySessionStore(), max_executions=2)

        initial = await system.start("Which option?", "A costs 10; B costs 20")
        self.assertEqual(initial.status, SessionStatus.WAITING_FOR_USER)
        self.assertEqual(initial.question.question, "What is your budget?")

        completed = await system.reply(initial.id, "My budget is 15")
        self.assertEqual(completed.status, SessionStatus.COMPLETED)
        self.assertEqual(completed.clarification_count, 1)
        self.assertEqual(completed.execution_count, 2)
        self.assertIn("My budget is 15", llm.prompts[3])
        self.assertIn("No reason given", llm.prompts[6])

    async def test_bounded_incomplete_result_preserves_critic_issues(self) -> None:
        llm = FakeLLM(
            [
                Classification(request_type="question", goal="Answer", ready_to_execute=True),
                ExecutionPlan(approach="Use context"),
                ExecutionResult(answer="Uncertain", gaps=["Missing source"]),
                Verification(
                    sufficient=False,
                    issues=["Missing source"],
                    follow_up_question="Can you provide a source?",
                ),
                FinalAnswer(answer="I cannot verify this yet."),
            ]
        )
        system = AgentSystem(llm, InMemorySessionStore(), max_executions=1)

        session = await system.start("What happened?")

        self.assertEqual(session.status, SessionStatus.COMPLETED)
        self.assertEqual(session.result.limitations, ["Missing source"])
        self.assertIsNone(session.question)

    async def test_failed_reply_keeps_session_waiting(self) -> None:
        llm = FakeLLM(
            [
                Classification(
                    request_type="question",
                    goal="Answer",
                    ready_to_execute=False,
                    missing_information=["detail"],
                ),
                SocraticQuestion(question="Which detail?", reason="Needed"),
            ]
        )
        system = AgentSystem(llm, InMemorySessionStore())
        initial = await system.start("Help me")

        with self.assertRaises(IndexError):
            await system.reply(initial.id, "The missing detail")

        restored = system.get(initial.id)
        self.assertEqual(restored.question.question, "Which detail?")
        self.assertEqual(restored.clarification_count, 0)
        self.assertEqual(len(restored.turns), len(initial.turns))
