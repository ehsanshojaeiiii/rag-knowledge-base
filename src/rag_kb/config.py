from functools import lru_cache

from openai.types.shared import ReasoningEffort
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables and `.env`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"

    embedding_base_url: str = "https://models-interview.arvancloudai.ir/v1"
    embedding_api_key: SecretStr | None = None
    embedding_model: str = "Bge-m3"

    # Any OpenAI-compatible chat endpoint; the default is the Ollama service from compose.yaml.
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: SecretStr = SecretStr("not-needed")  # local runners ignore it; hosted ones need it
    llm_model: str = "gemma4:e4b-it-qat"
    # Thinking models (Gemma 4) reason before answering; None sends nothing, leaving the model's
    # default (models without reasoning reject the parameter). compose.yaml sets "none" for Gemma.
    llm_reasoning_effort: ReasoningEffort = None

    # Below this cosine similarity a chunk is not treated as evidence. Calibrated for Bge-m3 by
    # eval/evaluate.py: the highest value that still refuses none of the answerable questions.
    min_similarity: float = 0.58

    max_upload_bytes: int = 20 * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
