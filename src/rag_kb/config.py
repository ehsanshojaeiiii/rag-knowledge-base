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

    # Any OpenAI-compatible chat endpoint; the default is a local Ollama.
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: SecretStr = SecretStr("ollama")  # Ollama ignores the key; hosted APIs need one
    llm_model: str = "gemma4"

    # Below this cosine similarity a chunk is not treated as evidence (calibrated by evaluation).
    min_similarity: float = 0.45

    max_upload_bytes: int = 20 * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
