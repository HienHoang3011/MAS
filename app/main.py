import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pymongo import AsyncMongoClient

from app.api.routers import health, v1
from app.config.config import Settings
from app.config.logging import configure_logging
from app.core.use_cases.agent_system import AgentSystem
from app.core.use_cases.chat_system import ChatSystem
from app.infra.clients.context_search import ContextSearchTool
from app.infra.db.agent_sessions import MongoAgentSessionStore
from app.infra.db.chat_store import MongoChatStore
from app.services.embedding_service import BgeM3Embedder
from app.services.llm_service import build_llm
from app.services.vector_store import BgeKnowledgeRetriever


def create_app() -> FastAPI:
    settings = Settings()  # reads env once
    configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if not settings.mongodb_uri:
            yield
            return
        client: AsyncMongoClient[dict[str, object]] = AsyncMongoClient(
            settings.mongodb_uri, serverSelectionTimeoutMS=15000, tz_aware=True
        )
        llm = build_llm(settings)
        stop = asyncio.Event()
        worker: asyncio.Task[None] | None = None
        try:
            store = MongoChatStore(client, settings.mongodb_database)
            await store.initialize()
            agent_store = MongoAgentSessionStore(client, settings.mongodb_database)
            await agent_store.initialize()
            app.state.agent_system = AgentSystem(
                llm,
                agent_store,
                tools=[ContextSearchTool()],
                max_clarifications=settings.agent_max_clarifications,
                max_executions=settings.agent_max_executions,
            )
            system = ChatSystem(
                llm,
                store,
                BgeKnowledgeRetriever(
                    client,
                    settings.mongodb_database,
                    BgeM3Embedder(settings.embedding_model, settings.embedding_cache_dir),
                ),
            )
            app.state.chat_system = system
            worker = asyncio.create_task(
                system.summary_worker(stop, settings.summary_poll_seconds),
                name="chat-memory-worker",
            )
            yield
        finally:
            stop.set()
            if worker is not None:
                worker.cancel()
                with suppress(asyncio.CancelledError):
                    await worker
            await llm.close()
            await client.close()

    app = FastAPI(
        title="My AI Project",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        root_path=settings.root_path,
        lifespan=lifespan,
    )

    origins = [o.strip() for o in settings.cors_allowed_origins.split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Routers
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(v1.router, prefix="/v1", tags=["v1"])

    return app


app = create_app()
