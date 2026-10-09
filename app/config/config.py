from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    env: str = Field(default="dev")
    log_level: str = Field(default="INFO")

    # LLM / Embeddings
    llm_provider: Literal["openai", "gemini"] = "openai"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash"
    agent_max_clarifications: int = Field(default=3, ge=0, le=10)
    agent_max_executions: int = Field(default=2, ge=1, le=10)

    # DB
    database_url: str | None = None  # e.g., postgres+psycopg://user:pass@host/db

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
