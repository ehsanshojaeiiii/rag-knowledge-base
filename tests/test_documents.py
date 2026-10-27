"""Document processing and change management, as required by the challenge.

Each test class maps to one requirement; the assertions check what is stored, because stored chunks
are the only thing answers can be built from.
"""

from typing import Any

import pytest
from conftest import FakeEmbedder
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_ingest import make_pdf

from rag_kb.config import get_settings
from rag_kb.models import Chunk


def upload(client: TestClient, filename: str, content: bytes, **form: str) -> Any:
    response = client.post("/documents", files={"file": (filename, content)}, data=form)
    assert response.status_code == 201, response.text
    return response.json()


def replace(client: TestClient, doc_id: str, filename: str, content: bytes, **headers: str) -> Any:
    return client.put(f"/documents/{doc_id}", files={"file": (filename, content)}, headers=headers)


def stored(db: Session, doc_id: str) -> list[tuple[int, str]]:
    """(version, content) of every chunk stored for a document."""
    rows = db.execute(
        select(Chunk.version, Chunk.content).where(Chunk.document_id == doc_id).order_by(Chunk.id)
    )
    return [(version, content) for version, content in rows]


class TestDocumentProcessing:
    """Requirement: text PDF, plain text and Markdown are processed and prepared for retrieval."""

    @pytest.mark.parametrize(
        ("filename", "content", "media_type"),
        [
            (
                "cdn.md",
                b"# CDN\n## Cache\nPurge the cache from the panel.",
                "text/markdown",
            ),
            ("notes.txt", "کش را از پنل پاک کنید.".encode(), "text/plain"),
            ("guide.pdf", make_pdf(["Purge the cache"]), "application/pdf"),
        ],
    )
    def test_supported_documents_are_stored_as_searchable_chunks(
        self, client: TestClient, db: Session, filename: str, content: bytes, media_type: str
    ) -> None:
        doc = upload(client, filename, content)

        assert (doc["media_type"], doc["version"]) == (media_type, 1)
        chunks = db.scalars(select(Chunk).where(Chunk.document_id == doc["id"])).all()
        assert chunks and all(c.version == 1 and c.embedding is not None for c in chunks)

    def test_markdown_sections_and_pdf_pages_are_kept_for_citations(
        self, client: TestClient, db: Session
    ) -> None:
        md = upload(client, "cdn.md", b"# CDN\n## Cache\nPurge the cache.")
        pdf = upload(client, "guide.pdf", make_pdf(["One", "Two"]))

        assert db.scalars(select(Chunk.section).where(Chunk.document_id == md["id"])).all() == [
            "CDN > Cache"
        ]
        assert db.scalars(select(Chunk.page).where(Chunk.document_id == pdf["id"])).all() == [1, 2]

    def test_title_defaults_to_file_name(self, client: TestClient) -> None:
        assert upload(client, "pricing.md", b"Prices")["title"] == "pricing"
        assert upload(client, "a.md", b"Prices", title="Price list")["title"] == "Price list"

    @pytest.mark.parametrize(
        ("filename", "content", "status"),
        [
            ("report.docx", b"binary", 415),
            ("broken.pdf", b"not a pdf", 422),
            ("empty.txt", b"   ", 422),
        ],
    )
    def test_unusable_files_are_rejected_with_a_clear_status(
        self, client: TestClient, filename: str, content: bytes, status: int
    ) -> None:
        response = client.post("/documents", files={"file": (filename, content)})

        assert response.status_code == status
        assert response.json()["detail"]
        assert client.get("/documents").json() == []

    def test_oversized_upload_is_rejected(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_settings(), "max_upload_bytes", 10)

        response = client.post("/documents", files={"file": ("big.txt", b"x" * 11)})

        assert response.status_code == 413


class TestChangeManagement:
    """Requirement: documents can be added, edited and deleted; outdated content is never used."""

    def test_edit_replaces_all_content_with_a_new_version(
        self, client: TestClient, db: Session
    ) -> None:
        doc = upload(client, "plan.md", b"The basic plan costs 10 dollars.")

        response = replace(client, doc["id"], "plan.md", b"The basic plan costs 12 dollars.")

        assert response.status_code == 200
        assert response.json()["version"] == 2
        assert stored(db, doc["id"]) == [(2, "The basic plan costs 12 dollars.")]

    def test_identical_upload_does_not_create_a_version(
        self, client: TestClient, embedder: FakeEmbedder
    ) -> None:
        doc = upload(client, "plan.md", b"Same content")
        calls = embedder.calls

        response = replace(client, doc["id"], "plan.md", b"Same content")

        assert response.json()["version"] == 1
        assert embedder.calls == calls

    def test_edit_based_on_an_outdated_version_is_rejected(
        self, client: TestClient, db: Session
    ) -> None:
        doc = upload(client, "plan.md", b"Version one")
        assert replace(client, doc["id"], "plan.md", b"Version two", **{"If-Match": "1"}).is_success

        response = replace(client, doc["id"], "plan.md", b"Lost update", **{"If-Match": "1"})

        assert response.status_code == 409
        assert stored(db, doc["id"]) == [(2, "Version two")]

    def test_failed_edit_keeps_the_previous_version_searchable(
        self, client: TestClient, db: Session, embedder: FakeEmbedder
    ) -> None:
        doc = upload(client, "plan.md", b"The basic plan costs 10 dollars.")
        embedder.fail = True

        response = replace(client, doc["id"], "plan.md", b"The basic plan costs 12 dollars.")

        assert response.status_code == 503
        assert client.get(f"/documents/{doc['id']}").json()["version"] == 1
        assert stored(db, doc["id"]) == [(1, "The basic plan costs 10 dollars.")]

    def test_delete_removes_the_document_and_all_its_chunks(
        self, client: TestClient, db: Session
    ) -> None:
        doc = upload(client, "plan.md", b"The basic plan costs 10 dollars.")

        assert client.delete(f"/documents/{doc['id']}").status_code == 204
        assert client.get(f"/documents/{doc['id']}").status_code == 404
        assert stored(db, doc["id"]) == []
        assert client.delete(f"/documents/{doc['id']}").status_code == 404

    def test_editing_a_missing_document_is_not_found(self, client: TestClient) -> None:
        missing = "01900000-0000-7000-8000-000000000000"
        assert replace(client, missing, "a.md", b"x").status_code == 404


class TestApiAccess:
    """Requirement: everything is reachable over a documented HTTP API."""

    def test_documents_are_listed_newest_first(self, client: TestClient) -> None:
        first = upload(client, "a.md", b"A")
        second = upload(client, "b.md", b"B")

        assert [d["id"] for d in client.get("/documents").json()] == [second["id"], first["id"]]

    def test_openapi_documents_every_endpoint(self, client: TestClient) -> None:
        paths = client.get("/openapi.json").json()["paths"]

        assert set(paths) >= {"/documents", "/documents/{document_id}", "/health"}
        assert set(paths["/documents/{document_id}"]) == {"get", "put", "delete"}
