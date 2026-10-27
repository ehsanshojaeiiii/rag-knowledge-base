from functools import lru_cache
from itertools import batched
from typing import Protocol

from openai import OpenAI, OpenAIError

from rag_kb.config import get_settings

# The provider accepts at most 200,000 characters per request;
# 64 chunks of at most 1,200 characters stay well below that.
BATCH_SIZE = 64


class EmbeddingUnavailable(RuntimeError):
    """The embedding service is not configured or did not answer."""


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAICompatibleEmbedder:
    """Embeds text through any OpenAI-compatible `/embeddings` endpoint.

    The SDK retries rate limits (429), server errors and timeouts with exponential backoff.
    """

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=60, max_retries=5)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        try:
            for batch in batched(texts, BATCH_SIZE, strict=False):
                response = self._client.embeddings.create(model=self._model, input=list(batch))
                vectors += [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
        except OpenAIError as exc:
            raise EmbeddingUnavailable(f"Embedding request failed: {type(exc).__name__}") from exc
        return vectors


@lru_cache
def get_embedder() -> Embedder:
    settings = get_settings()
    if settings.embedding_api_key is None:
        raise EmbeddingUnavailable("EMBEDDING_API_KEY is not set")
    return OpenAICompatibleEmbedder(
        settings.embedding_base_url,
        settings.embedding_api_key.get_secret_value(),
        settings.embedding_model,
    )
