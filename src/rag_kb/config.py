from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables and `.env`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"

    embedding_base_url: str = "https://models-interview.arvancloudai.ir/v1"
    embedding_api_key: SecretStr | None = None
    embedding_model: str = "Bge-m3"

    # Any OpenAI-compatible chat endpoint; the default is Docker Model Runner on this machine.
    llm_base_url: str = "http://localhost:12434/engines/v1"
    llm_api_key: SecretStr = SecretStr("not-needed")  # local runners ignore it; hosted ones need it
    llm_model: str = "ai/gemma4:e4b"

    # Below this cosine similarity a chunk is not treated as evidence (calibrated by evaluation).
    min_similarity: float = 0.45

    max_upload_bytes: int = 20 * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
