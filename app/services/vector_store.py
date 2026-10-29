"""Exact BGE-M3 dense retrieval for the small MongoDB textbook corpus."""

import hashlib
import logging

import numpy as np
from pymongo import AsyncMongoClient

from app.core.models.chat import EmbeddedChunk, KnowledgePassage, TextbookChunk
from app.services.embedding_service import BgeM3Embedder

logger = logging.getLogger(__name__)


class KnowledgeIndexNotReadyError(Exception):
    """The corpus must be indexed with the configured embedding model first."""


def _has_body(chunk: TextbookChunk) -> bool:
    # The source contains standalone headings that rank highly but provide no evidence.
    content = chunk.content.strip()
    return not any(
        content == " > ".join(chunk.heading_path[start:]).strip()
        for start in range(len(chunk.heading_path))
    )


class BgeKnowledgeRetriever:
    def __init__(
        self,
        client: AsyncMongoClient[dict[str, object]],
        database: str,
        embedder: BgeM3Embedder,
    ) -> None:
        self.collection = client[database]["knowledge_embeddings"]
        self.embedder = embedder

    async def search(self, query: str) -> list[KnowledgePassage]:
        docs = await self.collection.find({"model": self.embedder.model_name}).to_list()
        if not docs:
            raise KnowledgeIndexNotReadyError("Run make index-knowledge to build BGE-M3 embeddings")
        chunks = [chunk for doc in docs if _has_body(chunk := EmbeddedChunk.model_validate(doc))]
        if not chunks:
            raise ValueError("The textbook index contains only headings, without body text")
        vector = (await self.embedder.encode([query]))[0]
        matrix = np.asarray([chunk.embedding for chunk in chunks], dtype=np.float32)
        query_vector = np.asarray(vector, dtype=np.float32)
        if not np.isfinite(matrix).all() or not np.isfinite(query_vector).all():
            raise ValueError("Embedding vectors contain non-finite values")
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector)
        if np.any(norms == 0):
            raise ValueError("Embedding vectors must be nonzero")
        scores = matrix @ query_vector / norms
        indices = np.argsort(-scores, kind="stable")[:4]
        return [
            KnowledgePassage(
                id=chunks[i].chunk_id,
                content=chunks[i].content,
                source=" / ".join([chunks[i].source, *chunks[i].heading_path]),
                score=float(scores[i]),
            )
            for i in indices
        ]


async def index_textbook(
    client: AsyncMongoClient[dict[str, object]],
    source_database: str,
    source_collection: str,
    target_database: str,
    embedder: BgeM3Embedder,
) -> int:
    """Incrementally index existing chunks without changing the source collection."""
    target = client[target_database]["knowledge_embeddings"]
    await target.create_index([("chunk_id", 1), ("model", 1)], unique=True)
    count = 0
    docs = await client[source_database][source_collection].find().to_list()
    if not docs:
        raise ValueError("Textbook source collection is empty")
    pending: list[tuple[TextbookChunk, str]] = []
    for doc in docs:
        chunk = TextbookChunk.model_validate(doc)
        fingerprint = hashlib.sha256(chunk.model_dump_json().encode()).hexdigest()
        key = {"chunk_id": chunk.chunk_id, "model": embedder.model_name}
        existing = await target.find_one(key, {"content_hash": 1})
        if existing and existing.get("content_hash") == fingerprint:
            continue
        pending.append((chunk, fingerprint))
    for offset in range(0, len(pending), 8):
        batch = pending[offset : offset + 8]
        vectors = await embedder.encode([chunk.content for chunk, _ in batch])
        if len(vectors) != len(batch):
            raise ValueError("Embedding count does not match the textbook batch")
        for (chunk, fingerprint), vector in zip(batch, vectors, strict=True):
            embedded = EmbeddedChunk(
                **chunk.model_dump(),
                model=embedder.model_name,
                content_hash=fingerprint,
                embedding=vector,
            )
            await target.replace_one(
                {"chunk_id": chunk.chunk_id, "model": embedder.model_name},
                embedded.model_dump(),
                upsert=True,
            )
            count += 1
        logger.info("Indexed textbook chunks", extra={"indexed_count": count})
    return count
