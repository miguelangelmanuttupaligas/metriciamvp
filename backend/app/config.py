from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://dataai4:dataai4@localhost:5432/dataai4"
    storage_dir: Path = Path("/data")
    app_login_username: str | None = None
    app_login_password: str | None = None
    app_session_secret: str = "metricia-dev-session-secret"
    app_session_secure: bool = False
    app_session_max_age_seconds: int = 604800
    openai_api_key: str | None = None
    llm_model: str = "gpt-5"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    openai_timeout_seconds: float = 25.0
    ingestion_llm_analysis_json: bool = False
    row_chunk_size: int = 100

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    (settings.storage_dir / "uploads").mkdir(parents=True, exist_ok=True)
    (settings.storage_dir / "duckdb").mkdir(parents=True, exist_ok=True)
    return settings
