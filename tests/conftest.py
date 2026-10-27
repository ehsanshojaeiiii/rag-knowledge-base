from collections.abc import Iterator

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from testcontainers.community.postgres import PostgresContainer

from rag_kb.config import get_settings
from rag_kb.db import get_engine
from rag_kb.main import app

PGVECTOR_IMAGE = "pgvector/pgvector:0.8.7-pg18-trixie"


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


@pytest.fixture(scope="session")
def monkeypatch_session() -> Iterator[pytest.MonkeyPatch]:
    with pytest.MonkeyPatch.context() as mp:
        yield mp


@pytest.fixture
def client(database_url: str) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
