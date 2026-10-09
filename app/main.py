from fastapi import FastAPI

from app.api.routers import health, v1
from app.config.config import Settings
from app.config.logging import configure_logging


def create_app() -> FastAPI:
    settings = Settings()  # reads env once
    configure_logging(settings)

    app = FastAPI(
        title="My AI Project",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Routers
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(v1.router, prefix="/v1", tags=["v1"])

    return app


app = create_app()
