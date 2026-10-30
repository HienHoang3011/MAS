from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    env: str = Field(default="dev")
    log_level: str = Field(default="INFO")
    # Prefix khi chạy sau reverse proxy có strip path (vd /mas) — để /docs trỏ đúng openapi.json.
    root_path: str = ""
    # Danh sách origin được gọi API từ trình duyệt, phân cách bởi dấu phẩy. Trống = tắt CORS.
    cors_allowed_origins: str = ""

    # LLM / Embeddings
    llm_provider: Literal["openai", "gemini", "proxy"] = "openai"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash"
    agent_max_clarifications: int = Field(default=3, ge=0, le=10)
    agent_max_executions: int = Field(default=2, ge=1, le=10)
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
    llm_timeout: float = Field(default=120, gt=0, le=300)

    # DB
    database_url: str | None = None  # e.g., postgres+psycopg://user:pass@host/db
    mongodb_uri: str | None = None
    mongodb_database: str = "mas"
    mongodb_knowledge_database: str = "data"
    mongodb_knowledge_collection: str = "gtrinh"
    embedding_model: str = "BAAI/bge-m3"
    embedding_cache_dir: str = ".cache/models"
    summary_poll_seconds: float = Field(default=5, gt=0)

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
