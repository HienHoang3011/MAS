"""Verify ranking at the embedding/database boundary using deterministic vectors."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from app.services import vector_store
from app.services.embedding_service import BgeM3Embedder
from app.services.vector_store import BgeKnowledgeRetriever, KnowledgeIndexNotReadyError


def test_bge_retrieval_ranks_cosine_scores_and_returns_source_metadata() -> None:
    client = MagicMock()
    embedder = Mock(spec=BgeM3Embedder)
    embedder.model_name = "BAAI/bge-m3"
    embedder.encode = AsyncMock(return_value=[[1.0, 0.0] + [0.0] * 1022])
    retriever = BgeKnowledgeRetriever(client, "test", embedder)
    docs = [
        {
            "chunk_id": str(i),
            "source": "gtrinh",
            "heading_path": ["Định lý bắt tay"],
            "content": "Tổng bậc của đồ thị.",
            "content_hash": "hash",
            "model": "BAAI/bge-m3",
            "embedding": vector + [0.0] * 1022,
        }
        for i, vector in enumerate([[1.0, 0.0], [5.0, 5.0], [0.0, 1.0], [-1.0, 1.0], [-1.0, 0.0]])
    ]
    client["test"]["knowledge_embeddings"].find.return_value.to_list = AsyncMock(return_value=docs)
    passages = asyncio.run(retriever.search("Định lý bắt tay"))
    assert [p.id for p in passages] == ["0", "1", "2", "3"]
    assert passages[0].score == pytest.approx(1.0)
    assert passages[1].score == pytest.approx(2**-0.5)
    assert passages[0].source == "gtrinh / Định lý bắt tay"
    embedder.encode.assert_awaited_once_with(["Định lý bắt tay"])


def test_missing_index_fails_explicitly_before_loading_embedding_model() -> None:
    client = MagicMock()
    embedder = Mock(spec=BgeM3Embedder)
    embedder.model_name = "BAAI/bge-m3"
    retriever = BgeKnowledgeRetriever(client, "test", embedder)
    client["test"]["knowledge_embeddings"].find.return_value.to_list = AsyncMock(return_value=[])
    with pytest.raises(KnowledgeIndexNotReadyError, match="index-knowledge"):
        asyncio.run(retriever.search("Dijkstra"))
    embedder.encode.assert_not_called()


def test_heading_only_chunks_do_not_displace_actual_theory() -> None:
    client = MagicMock()
    embedder = Mock(spec=BgeM3Embedder)
    embedder.model_name = "BAAI/bge-m3"
    embedder.encode = AsyncMock(return_value=[[1.0, 0.0] + [0.0] * 1022])
    retriever = BgeKnowledgeRetriever(client, "test", embedder)
    docs = [
        {
            "chunk_id": "heading",
            "source": "gtrinh",
            "heading_path": ["Chương 7", "Định nghĩa cây"],
            "content": "Chương 7 > Định nghĩa cây\n\n",
            "content_hash": "heading-hash",
            "model": "BAAI/bge-m3",
            "embedding": [1.0, 0.0] + [0.0] * 1022,
        },
        {
            "chunk_id": "definition",
            "source": "gtrinh",
            "heading_path": ["Chương 7", "Định nghĩa cây"],
            "content": "Định nghĩa cây\n\nCây là đồ thị liên thông không có chu trình.",
            "content_hash": "definition-hash",
            "model": "BAAI/bge-m3",
            "embedding": [1.0, 0.1] + [0.0] * 1022,
        },
    ]
    client["test"]["knowledge_embeddings"].find.return_value.to_list = AsyncMock(return_value=docs)
    passages = asyncio.run(retriever.search("Cây là gì?"))
    assert [passage.id for passage in passages] == ["definition"]


def test_index_cache_shares_concurrent_load_and_refreshes_after_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    monkeypatch.setattr(vector_store, "monotonic", lambda: clock[0])
    embedder = Mock(spec=BgeM3Embedder)
    embedder.model_name = "BAAI/bge-m3"
    embedder.encode = AsyncMock(return_value=[[1.0] + [0.0] * 1023])
    client = MagicMock()
    retriever = BgeKnowledgeRetriever(client, "test", embedder)
    doc = {
        "chunk_id": "tree",
        "source": "textbook",
        "content": "Original definition.",
        "content_hash": "hash",
        "model": "BAAI/bge-m3",
        "embedding": [1.0] + [0.0] * 1023,
    }
    read = AsyncMock(return_value=[doc])
    client["test"]["knowledge_embeddings"].find.return_value.to_list = read

    async def scenario() -> None:
        first, second = await asyncio.gather(retriever.search("tree"), retriever.search("graph"))
        assert first[0].score == second[0].score == pytest.approx(1)
        read.assert_awaited_once()
        read.return_value = [{**doc, "content": "Updated definition."}]
        clock[0] = 299
        assert (await retriever.search("tree"))[0].content == "Original definition."
        read.assert_awaited_once()
        clock[0] = 301
        assert (await retriever.search("tree"))[0].content == "Updated definition."
        assert read.await_count == 2

    asyncio.run(scenario())
