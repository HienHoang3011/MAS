"""Test routing, pedagogical review and memory boundaries without network calls."""

import asyncio
import json
from collections import deque
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import BaseModel, ValidationError

from app.core.models.chat import (
    ChatMessage,
    ChatRequest,
    ChatRoute,
    ChatSession,
    FactReview,
    GuidanceReview,
    KnowledgePassage,
    MemorySummary,
    ReplyCoverage,
    ReplyReview,
    RequirementCoverage,
    RoutingDecision,
    SourceCitation,
)
from app.core.ports.llm import T
from app.core.use_cases.chat_system import ChatSystem, GuidancePolicyError, text_blocks
from app.infra.db.chat_store import MongoChatStore
from app.services.vector_store import BgeKnowledgeRetriever


class ScriptedLLM:
    def __init__(self, responses: list[BaseModel | str]) -> None:
        self.responses = deque(responses)
        self.inputs: list[dict[str, object] | list[dict[str, object]]] = []

    async def complete(self, system: str, user: str, response_model: type[T]) -> T:
        self.inputs.append(json.loads(user))
        response = self.responses.popleft()
        assert isinstance(response, BaseModel)
        return response_model.model_validate(response.model_dump())

    async def generate(self, system: str, user: str) -> str:
        self.inputs.append(json.loads(user))
        response = self.responses.popleft()
        assert isinstance(response, str)
        return response


def approved_review(quote: str, passage: KnowledgePassage | None = None) -> GuidanceReview:
    return GuidanceReview(
        citations=[
            SourceCitation(
                answer_block_ids=[0] if quote.strip() else [],
                source_index=0,
                source_block_ids=[0],
            )
        ]
        if passage
        else [],
        grounded=True,
        pedagogically_safe=True,
        feedback="",
    )


def approved_reply_review(answer: str) -> ReplyReview:
    return ReplyReview(
        checks=[
            RequirementCoverage(
                requirement_index=0,
                satisfied=True,
                answer_block_ids=[0] if answer.strip() else [],
                issue=None,
            )
        ],
        facts=[
            FactReview(
                block_id=i,
                claim=text_blocks(answer)[i],
                assessment="Correct statement.",
                correct=True,
                issue=None,
            )
            for i in range(len(text_blocks(answer)))
        ],
        feedback="",
    )


def dependencies() -> tuple[Mock, Mock]:
    store = Mock(spec=MongoChatStore)
    store.resolve_session = AsyncMock(return_value=ChatSession(user_id="student"))
    store.recent_messages = AsyncMock(return_value=[])
    store.recall = AsyncMock(return_value=[])
    store.commit_turn = AsyncMock()
    retriever = Mock(spec=BgeKnowledgeRetriever)
    retriever.search = AsyncMock(return_value=[])
    return store, retriever


def test_chat_skips_retrieval_for_casual_conversation() -> None:
    store, retriever = dependencies()
    llm = ScriptedLLM(
        [
            RoutingDecision(route=ChatRoute.CONVERSATION, requirements=["Respond naturally"]),
            "Chào bạn!",
            approved_review("Chào bạn!"),
        ]
    )
    response = asyncio.run(ChatSystem(llm, store, retriever).chat(ChatRequest(content="Xin chào")))
    assert response.answer == "Chào bạn!"
    assert response.session_id == store.resolve_session.return_value.id
    retriever.search.assert_not_called()
    store.commit_turn.assert_awaited_once()


def test_fast_chat_calls_only_generation_and_saves_the_turn() -> None:
    store, retriever = dependencies()
    llm = ScriptedLLM(["Hãy bắt đầu bằng cách xác định các đỉnh kề của A."])
    response = asyncio.run(
        ChatSystem(llm, store, retriever).chat(
            ChatRequest(content="Hướng dẫn mình làm BFS", mode="fast")
        )
    )
    assert response.route == ChatRoute.DIRECT
    assert response.sources == []
    assert response.answer == "Hãy bắt đầu bằng cách xác định các đỉnh kề của A."
    assert len(llm.inputs) == 1
    assert llm.inputs[0]["content"] == "Hướng dẫn mình làm BFS"
    retriever.search.assert_not_called()
    store.commit_turn.assert_awaited_once_with(
        store.resolve_session.return_value,
        "Hướng dẫn mình làm BFS",
        response.answer,
    )


def test_auto_chat_answers_simple_question_in_one_llm_call() -> None:
    store, retriever = dependencies()
    llm = ScriptedLLM(
        [
            RoutingDecision(
                route=ChatRoute.CONVERSATION,
                requirements=[],
                use_fast=True,
                fast_answer="Chào bạn!",
            )
        ]
    )
    response = asyncio.run(ChatSystem(llm, store, retriever).chat(ChatRequest(content="Xin chào")))
    assert response.route == ChatRoute.DIRECT
    assert response.answer == "Chào bạn!"
    assert len(llm.inputs) == 1
    retriever.search.assert_not_called()
    store.commit_turn.assert_awaited_once()


def test_auto_chat_keeps_exercise_in_reviewed_flow() -> None:
    store, retriever = dependencies()
    answer = "Hãy bắt đầu bằng cách đếm bậc từng đỉnh."
    llm = ScriptedLLM(
        [
            RoutingDecision(
                route=ChatRoute.TUTORING,
                requirements=["Give a hint"],
                use_fast=True,
                fast_answer="Đáp án cuối.",
            ),
            answer,
            approved_reply_review(answer),
            approved_review(answer),
        ]
    )
    response = asyncio.run(
        ChatSystem(llm, store, retriever).chat(ChatRequest(content="Giải bài đồ thị này"))
    )
    assert response.route == ChatRoute.TUTORING
    assert response.answer == answer
    assert len(llm.inputs) == 4
    retriever.search.assert_awaited_once()


def test_router_cannot_select_direct_mode() -> None:
    with pytest.raises(ValidationError):
        RoutingDecision(route=ChatRoute.DIRECT, requirements=["Respond"])


def test_tutoring_rewrites_a_leaked_answer_before_saving() -> None:
    store, retriever = dependencies()
    retriever.search.return_value = [
        KnowledgePassage(
            id="handshake", source="Giáo trình", content="Tổng bậc bằng 2|E|.", score=0.8
        )
    ]
    llm = ScriptedLLM(
        [
            RoutingDecision(
                route=ChatRoute.TUTORING,
                retrieval_query="định lý bắt tay",
                requirements=["Explain the method without giving the edge count"],
            ),
            "Có 3 cạnh.",
            approved_reply_review("Có 3 cạnh."),
            approved_review("Có 3 cạnh.").model_copy(
                update={
                    "pedagogically_safe": False,
                    "feedback": "Do not give the final edge count.",
                }
            ),
            "Hãy cộng các bậc rồi áp dụng định lý bắt tay.",
            approved_reply_review("Hãy cộng các bậc rồi áp dụng định lý bắt tay."),
            approved_review(
                "Hãy cộng các bậc rồi áp dụng định lý bắt tay.",
                retriever.search.return_value[0],
            ),
        ]
    )
    response = asyncio.run(
        ChatSystem(llm, store, retriever).chat(
            ChatRequest(content="Tính số cạnh khi các bậc là 2, 2, 2", user_id="student")
        )
    )
    assert "3 cạnh" not in response.answer
    assert response.sources[0].id == "handshake"
    assert "content" not in response.sources[0].model_dump()
    store.commit_turn.assert_awaited_once_with(
        store.resolve_session.return_value,
        "Tính số cạnh khi các bậc là 2, 2, 2",
        response.answer,
    )
    assert "Do not give the final edge count." in json.dumps(llm.inputs[4])


def test_rejected_replies_are_not_persisted() -> None:
    store, retriever = dependencies()
    llm = ScriptedLLM(
        [
            RoutingDecision(route=ChatRoute.TUTORING, requirements=["Give a next-step hint"]),
            "Đáp án cuối.",
            approved_reply_review("Đáp án cuối."),
            approved_review("Đáp án cuối.").model_copy(
                update={"pedagogically_safe": False, "feedback": "Leaks answer"}
            ),
            "Đáp án cuối.",
            approved_reply_review("Đáp án cuối."),
            approved_review("Đáp án cuối.").model_copy(
                update={"pedagogically_safe": False, "feedback": "Still leaks"}
            ),
        ]
    )
    with pytest.raises(GuidancePolicyError):
        asyncio.run(ChatSystem(llm, store, retriever).chat(ChatRequest(content="Giải bài này")))
    store.commit_turn.assert_not_called()


def test_summary_archives_only_messages_outside_recent_ten() -> None:
    store, retriever = dependencies()
    session = ChatSession(user_id="student", message_count=14)
    store.claim_summary = AsyncMock(return_value=session)
    now = datetime.now(UTC)
    store.message_range = AsyncMock(
        return_value=[
            ChatMessage(
                session_id=session.id,
                user_id=session.user_id,
                sequence=i,
                role="user" if i % 2 else "assistant",
                content=f"message {i}",
                created_at=now,
            )
            for i in range(1, 5)
        ]
    )
    store.save_summary = AsyncMock()
    llm = ScriptedLLM([MemorySummary(summary="Người học đang tìm hiểu bậc của đỉnh.")])
    assert asyncio.run(ChatSystem(llm, store, retriever).summarize_next())
    store.message_range.assert_awaited_once_with(session, 0, 4)
    memory = store.save_summary.call_args.args[1]
    assert (memory.from_sequence, memory.through_sequence) == (1, 4)
    assert memory.user_id == "student" and memory.session_id == session.id


def test_invalid_summary_stops_retries_without_advancing_memory() -> None:
    store, retriever = dependencies()
    session = ChatSession(user_id="student", message_count=12)
    store.claim_summary = AsyncMock(return_value=session)
    store.message_range = AsyncMock(return_value=[])
    store.fail_summary = AsyncMock()
    store.save_summary = AsyncMock()
    asyncio.run(ChatSystem(ScriptedLLM([]), store, retriever).summarize_next())
    store.fail_summary.assert_awaited_once_with(session, "ValueError", None)
    store.save_summary.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [
        {"content": "  "},
        {"content": "hello", "user_id": ""},
        {"content": "hello", "session_id": "session"},
        {"content": "hello", "mode": "unknown"},
    ],
)
def test_invalid_chat_identity_or_content_is_rejected(payload: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(payload)


def test_nonexistent_evidence_cannot_approve_an_introduction() -> None:
    store, retriever = dependencies()
    passage = KnowledgePassage(
        id="tree",
        source="Textbook",
        score=0.9,
        content="Cây là đồ thị vô hướng liên thông không có chu trình.",
    )
    retriever.search.return_value = [passage]
    introduction = "Dưới đây là định nghĩa chi tiết về cây."
    definition = passage.content
    llm = ScriptedLLM(
        [
            RoutingDecision(route=ChatRoute.KNOWLEDGE, requirements=["Define a tree"]),
            introduction,
            approved_reply_review(introduction),
            approved_review(definition, passage).model_copy(
                update={
                    "citations": [
                        approved_review(definition, passage)
                        .citations[0]
                        .model_copy(update={"answer_block_ids": [99]})
                    ]
                }
            ),
            definition,
            approved_reply_review(definition),
            approved_review(definition, passage),
        ]
    )
    response = asyncio.run(
        ChatSystem(llm, store, retriever).chat(ChatRequest(content="Cây là gì?"))
    )
    assert response.answer == definition
    rewrite = llm.inputs[4]
    assert isinstance(rewrite, dict)
    assert rewrite["previous_draft"] == introduction
    assert rewrite["review_feedback"]
    store.commit_turn.assert_awaited_once_with(
        store.resolve_session.return_value, "Cây là gì?", definition
    )


@pytest.mark.parametrize("defect", ["missing_requirement", "duplicate_requirement"])
def test_reply_review_requires_complete_coverage(defect: str) -> None:
    from app.core.use_cases.chat_system import reply_review_issues

    answer = "Cây là đồ thị liên thông không có chu trình. Cây có n-1 cạnh."
    routing = RoutingDecision(
        route=ChatRoute.KNOWLEDGE, requirements=["Define a tree", "Explain its edge count"]
    )
    review = approved_reply_review(answer)
    review.checks.append(review.checks[0].model_copy(update={"requirement_index": 1}))
    assert not reply_review_issues(review, routing, answer)
    if defect == "missing_requirement":
        review.checks.pop()
    else:
        review.checks[1] = review.checks[0]
    assert reply_review_issues(review, routing, answer)


def test_citation_cannot_reference_absent_evidence() -> None:
    from app.core.use_cases.chat_system import review_issues

    passage = KnowledgePassage(id="tree", source="Textbook", score=0.9, content="Cây có n-1 cạnh.")
    routing = RoutingDecision(route=ChatRoute.KNOWLEDGE, requirements=["Explain its edge count"])
    review = approved_review(passage.content, passage)
    assert not review_issues(review, routing, passage.content, [passage])
    review.citations[0].source_block_ids = [99]
    assert review_issues(review, routing, passage.content, [passage])


def test_unverifiable_reviews_exhaust_attempts_without_saving() -> None:
    store, retriever = dependencies()
    llm = ScriptedLLM(
        [
            RoutingDecision(route=ChatRoute.KNOWLEDGE, requirements=["Explain the concept"]),
            "I will greet you.",
            approved_reply_review("Hello!").model_copy(
                update={
                    "facts": [
                        FactReview(
                            block_id=0,
                            claim="Candidate statement.",
                            assessment="The stated rule contradicts this claim.",
                            correct=False,
                            issue="Invalid statement.",
                        )
                    ]
                }
            ),
            "I will respond now.",
            approved_reply_review("Hello!").model_copy(
                update={
                    "facts": [
                        FactReview(
                            block_id=0,
                            claim="Candidate statement.",
                            assessment="The stated rule contradicts this claim.",
                            correct=False,
                            issue="Invalid statement.",
                        )
                    ]
                }
            ),
        ]
    )
    with pytest.raises(GuidancePolicyError):
        asyncio.run(ChatSystem(llm, store, retriever).chat(ChatRequest(content="Hi")))
    store.commit_turn.assert_not_called()


def test_incomplete_reply_is_rewritten_before_reading_sources() -> None:
    store, retriever = dependencies()
    passage = KnowledgePassage(
        id="tree",
        source="Textbook",
        score=0.9,
        content="Cây là đồ thị vô hướng liên thông không có chu trình.",
    )
    retriever.search.return_value = [passage]
    llm = ScriptedLLM(
        [
            RoutingDecision(route=ChatRoute.KNOWLEDGE, requirements=["Define a tree"]),
            "Sau đây là định nghĩa cây.",
            ReplyReview(
                checks=[
                    RequirementCoverage(
                        requirement_index=0,
                        satisfied=False,
                        answer_block_ids=[],
                        issue="The definition is missing.",
                    )
                ],
                facts=[
                    FactReview(
                        block_id=0,
                        claim="Introductory statement.",
                        assessment="No factual error.",
                        correct=True,
                        issue=None,
                    )
                ],
                feedback="Provide the definition, not a promise.",
            ),
            passage.content,
            approved_reply_review(passage.content),
            approved_review(passage.content, passage),
        ]
    )
    response = asyncio.run(
        ChatSystem(llm, store, retriever).chat(ChatRequest(content="Cây là gì?"))
    )
    assert response.answer == passage.content
    assert isinstance(llm.inputs[2], dict)
    assert "retrieved_passages" not in llm.inputs[2]
    assert "retrieved_passages" in llm.inputs[-1]
    store.commit_turn.assert_awaited_once()


def test_inaccurate_reply_is_rewritten_before_source_review() -> None:
    store, retriever = dependencies()
    corrected = "Cạnh âm có thể làm đường đi ngắn hơn."
    llm = ScriptedLLM(
        [
            RoutingDecision(route=ChatRoute.KNOWLEDGE, requirements=["Explain negative edges"]),
            "Cạnh âm làm đường đi dài hơn.",
            approved_reply_review("Cạnh âm làm đường đi dài hơn.").model_copy(
                update={
                    "facts": [
                        FactReview(
                            block_id=0,
                            claim="Negative weights increase distance.",
                            assessment="Negative weights decrease distance.",
                            correct=False,
                            issue="Negative weights reduce distance.",
                        )
                    ],
                    "feedback": "Negative weights reduce distance.",
                }
            ),
            corrected,
            approved_reply_review(corrected),
            approved_review(corrected),
        ]
    )
    response = asyncio.run(
        ChatSystem(llm, store, retriever).chat(ChatRequest(content="Cạnh âm ảnh hưởng gì?"))
    )
    assert response.answer == corrected
    assert "Negative weights reduce distance." in json.dumps(llm.inputs[3])
    store.commit_turn.assert_awaited_once_with(
        store.resolve_session.return_value, "Cạnh âm ảnh hưởng gì?", corrected
    )


def test_reply_review_cannot_skip_a_factual_statement() -> None:
    from app.core.use_cases.chat_system import reply_review_issues

    answer = "Thuật toán chọn đỉnh có nhãn nhỏ nhất. Cạnh âm có thể giảm khoảng cách."
    routing = RoutingDecision(route=ChatRoute.KNOWLEDGE, requirements=["Explain the algorithm"])
    review = approved_reply_review(answer)
    assert not reply_review_issues(review, routing, answer)
    review.facts.pop()
    assert reply_review_issues(review, routing, answer)


def test_next_step_question_can_share_the_methods_source() -> None:
    from app.core.use_cases.chat_system import review_issues

    passage = KnowledgePassage(
        id="handshake", source="Textbook", score=0.9, content="Tổng bậc bằng 2m."
    )
    answer = "Tổng bậc bằng 2m. Bạn hãy cộng các bậc đã cho nhé?"
    routing = RoutingDecision(
        route=ChatRoute.TUTORING, requirements=["Explain the method", "Ask the next step"]
    )
    review = approved_review(answer, passage)
    assert not review_issues(review, routing, answer, [passage])


def test_tutoring_accepts_instructions_without_factual_claims() -> None:
    from app.core.use_cases.chat_system import coverage_issues

    answer = "Áp dụng định lý tổng bậc rồi thử cộng các bậc đã cho nhé?"
    review = ReplyCoverage(checks=approved_reply_review(answer).checks)
    assert not coverage_issues(review.checks, ["Give the next action"], answer)


def test_citation_can_cover_more_than_eight_real_blocks() -> None:
    from app.core.use_cases.chat_system import review_issues

    answer = "\n".join(f"Statement {i}." for i in range(12))
    passage = KnowledgePassage(id="long", source="Textbook", score=0.9, content=answer)
    review = approved_review(answer, passage)
    review.citations[0] = SourceCitation(
        answer_block_ids=list(range(12)), source_index=0, source_block_ids=list(range(12))
    )
    routing = RoutingDecision(route=ChatRoute.KNOWLEDGE, requirements=["Explain the properties"])
    assert not review_issues(review, routing, answer, [passage])
