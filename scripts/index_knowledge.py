"""Build resumable BGE-M3 embeddings for the configured MongoDB textbook."""

import asyncio
import logging

from pymongo import AsyncMongoClient

from app.config.config import get_settings
from app.services.embedding_service import BgeM3Embedder
from app.services.vector_store import index_textbook


async def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=logging.INFO)
    if not settings.mongodb_uri:
        raise ValueError("MONGODB_URI is not configured")
    client: AsyncMongoClient[dict[str, object]] = AsyncMongoClient(
        settings.mongodb_uri, serverSelectionTimeoutMS=15000, tz_aware=True
    )
    try:
        count = await index_textbook(
            client,
            settings.mongodb_knowledge_database,
            settings.mongodb_knowledge_collection,
            settings.mongodb_database,
            BgeM3Embedder(settings.embedding_model, settings.embedding_cache_dir),
        )
        print(f"Indexed {count} new or changed chunks with BGE-M3.")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
