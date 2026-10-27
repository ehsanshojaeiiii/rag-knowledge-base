"""Retrieval, as required by the challenge: relevant chunks are found, outdated ones never are."""

from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from test_documents import replace, upload

from conftest import FakeEmbedder
from rag_kb.models import Chunk
from rag_kb.retrieval import keyword_query


def search(client: TestClient, query: str, top_k: int = 5) -> list[Any]:
    response = client.post("/search", json={"query": query, "top_k": top_k})
    assert response.status_code == 200, response.text
    hits: list[Any] = response.json()
    return hits


class TestRelevantRetrieval:
    """Requirement: identify the parts of the documents related to the question."""

    def test_the_chunk_that_answers_ranks_first(self, client: TestClient) -> None:
        upload(client, "cdn.md", b"# CDN\nPurge the CDN cache from the panel settings.")
        upload(client, "billing.md", b"# Billing\nInvoices are issued on the first day of a month.")

        hits = search(client, "How do I purge the CDN cache?")

        assert hits[0]["title"] == "cdn"

    def test_exact_identifiers_are_found(self, client: TestClient) -> None:
        upload(client, "errors.md", b"Error ERR_4021 means the request quota is exhausted.")
        upload(client, "other.md", b"Requests are limited per minute for every account.")

        assert search(client, "what is ERR_4021")[0]["title"] == "errors"

    def test_half_space_spellings_match(self, client: TestClient) -> None:
        upload(client, "fa.md", "پاک‌سازی کش از پنل انجام می‌شود".encode())
        upload(client, "other.md", "صورت‌حساب ماهانه صادر می‌شود".encode())

        assert search(client, "پاک سازی کش")[0]["title"] == "fa"

    def test_hits_carry_what_a_citation_needs(self, client: TestClient) -> None:
        doc = upload(client, "cdn.md", b"# CDN\n## Cache\nPurge the cache from the panel.")

        [hit] = search(client, "purge cache", top_k=1)

        assert hit["document_id"] == doc["id"]
        assert (hit["version"], hit["section"], hit["page"]) == (1, "CDN > Cache", None)
        assert "Purge the cache" in hit["content"]
        assert hit["similarity"] is not None

    def test_top_k_limits_the_number_of_hits(self, client: TestClient) -> None:
        for i in range(4):
            upload(client, f"doc{i}.md", f"Cache rule number {i}".encode())

        assert len(search(client, "cache rule", top_k=2)) == 2

    def test_invalid_queries_are_rejected(self, client: TestClient) -> None:
        assert client.post("/search", json={"query": ""}).status_code == 422
        assert client.post("/search", json={"query": "x", "top_k": 50}).status_code == 422


class TestOutdatedContentIsNeverRetrieved:
    """Requirement: after a change, outdated content must not be the basis of new answers."""

    def test_only_the_current_version_is_retrieved_after_an_edit(self, client: TestClient) -> None:
        doc = upload(client, "plan.md", b"The basic plan costs 10 dollars per month.")
        replace(client, doc["id"], "plan.md", b"The basic plan costs 12 dollars per month.")

        hits = search(client, "basic plan cost")

        assert [(h["version"], h["content"]) for h in hits] == [
            (2, "The basic plan costs 12 dollars per month.")
        ]

    def test_stale_chunks_are_excluded_even_if_they_were_not_cleaned_up(
        self, client: TestClient, db: Session
    ) -> None:
        doc = upload(client, "plan.md", b"The basic plan costs 10 dollars per month.")
        replace(client, doc["id"], "plan.md", b"The basic plan costs 12 dollars per month.")
        old = "The basic plan costs 10 dollars per month."
        db.add(
            Chunk(
                document_id=doc["id"],
                version=1,
                ordinal=0,
                content=old,
                embedding=FakeEmbedder().embed([old])[0],
            )
        )
        db.commit()

        assert all(h["version"] == 2 for h in search(client, "basic plan 10 dollars"))

    def test_deleted_documents_are_never_retrieved(self, client: TestClient) -> None:
        doc = upload(client, "plan.md", b"The basic plan costs 10 dollars per month.")
        client.delete(f"/documents/{doc['id']}")

        assert search(client, "basic plan cost") == []


class TestKeywordQuery:
    def test_drops_common_words_and_splits_like_postgres(self) -> None:
        assert keyword_query("چطور کش CDN را پاک‌سازی کنم؟ ERR_4021") == (
            "کش | cdn | پاک | سازی | err | 4021"
        )

    def test_a_question_of_only_common_words_has_no_terms(self) -> None:
        assert keyword_query("how is it?") == ""
