import zlib
from collections.abc import Iterator

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session
from testcontainers.community.postgres import PostgresContainer

from rag_kb.answering import LLMUnavailable, get_llm
from rag_kb.config import get_settings
from rag_kb.db import get_engine
from rag_kb.embeddings import EmbeddingUnavailable, get_embedder
from rag_kb.ingest import normalize
from rag_kb.main import app
from rag_kb.models import EMBEDDING_DIMENSIONS

PGVECTOR_IMAGE = "pgvector/pgvector:0.8.7-pg18-trixie"


class FakeEmbedder:
    """Deterministic stand-in for the embedding API: texts sharing words get similar vectors."""

    def __init__(self) -> None:
        self.calls = 0
        self.fail = False

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.fail:
            raise EmbeddingUnavailable("Embedding request failed: APITimeoutError")
        return [self._vector(t) for t in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        vector = [0.0] * EMBEDDING_DIMENSIONS
        for word in normalize(text).lower().split():
            vector[zlib.crc32(word.encode()) % EMBEDDING_DIMENSIONS] += 1.0
        vector[0] += 0.01  # never all zeros: cosine distance is undefined for a zero vector
        return vector


class FakeLLM:
    """Stand-in for the chat model: returns a scripted reply and records every prompt it gets."""

    def __init__(self) -> None:
        self.reply = "Grounded answer [1]."
        self.prompts: list[str] = []
        self.fail = False

    def complete(self, system: str, user: str) -> str:
        self.prompts.append(user)
        if self.fail:
            raise LLMUnavailable("Language model request failed: APIConnectionError")
        return self.reply


@pytest.fixture(scope="session")
def monkeypatch_session() -> Iterator[pytest.MonkeyPatch]:
    with pytest.MonkeyPatch.context() as mp:
        yield mp


@pytest.fixture(scope="session")
def database_url(monkeypatch_session: pytest.MonkeyPatch) -> Iterator[str]:
    """A throwaway Postgres with pgvector, migrated to the latest schema."""
    with PostgresContainer(PGVECTOR_IMAGE, driver="psycopg") as postgres:
        url = postgres.get_connection_url()
        monkeypatch_session.setenv("DATABASE_URL", url)
        get_settings.cache_clear()
        get_engine.cache_clear()
        command.upgrade(Config("alembic.ini"), "head")
        yield url
        get_engine().dispose()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def client(database_url: str, embedder: FakeEmbedder, llm: FakeLLM) -> Iterator[TestClient]:
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_llm] = lambda: llm
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    with get_engine().begin() as connection:
        connection.execute(text("TRUNCATE documents CASCADE"))


@pytest.fixture
def db(database_url: str) -> Iterator[Session]:
    """A session for asserting on what is stored."""
    with Session(get_engine()) as session:
        yield session
