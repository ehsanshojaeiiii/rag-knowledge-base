"""Find the chunks most relevant to a question, from current versions of existing documents only.

Two searches run in one query and are fused by rank (Reciprocal Rank Fusion):
- semantic: nearest embeddings (cosine), good at paraphrases and cross-language questions;
- keyword: Postgres full-text match, good at exact terms such as product names and error codes.
"""

import re
from dataclasses import dataclass, replace
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from rag_kb.embeddings import Embedder
from rag_kb.ingest import normalize

CANDIDATES = 20  # taken from each search before fusion
RRF_K = 60  # standard constant from the RRF paper; damps the weight of top ranks

# Words too common to say anything about relevance. Postgres ranking has no notion of word
# frequency, so a question's "را" or "how" would otherwise match almost every chunk.
STOPWORDS = frozenset(
    "از به با در را و که این آن برای تا یا هم اما اگر چه چی چطور چگونه چرا کجا کی چند آیا "  # noqa: SIM905
    "است هست بود شد شود می نمی کنم کنیم کنید کند کرد کردن دارد دارم داریم باید توان "
    "من ما شما او آنها یک هر همه چیست کدام "
    "a an the is are was were be to of in on for and or how what why when where which who "
    "do does did can could should i we you it this that with from by as at".split()
)

# Each search reads `chunks` directly, so the HNSW and GIN indexes can serve it, and keeps only
# chunks of the current version through the join. Ranks are numbered after LIMIT, so neither
# search sorts the whole table.
_SEARCH = text("""
    WITH semantic AS (
        SELECT id, row_number() OVER (ORDER BY distance) AS rank
        FROM (
            SELECT c.id, c.embedding <=> CAST(:embedding AS vector) AS distance
            FROM chunks c
            JOIN documents d ON d.id = c.document_id AND d.version = c.version
            ORDER BY distance
            LIMIT :candidates
        ) nearest
    ),
    keyword AS (
        SELECT id, row_number() OVER (ORDER BY relevance DESC) AS rank
        FROM (
            SELECT c.id, ts_rank_cd(c.search, query) AS relevance
            FROM chunks c
            JOIN documents d ON d.id = c.document_id AND d.version = c.version,
                 to_tsquery('simple', :terms) AS query
            WHERE :terms <> '' AND c.search @@ query
            ORDER BY relevance DESC
            LIMIT :candidates
        ) matching
    )
    SELECT c.id AS chunk_id, c.document_id, d.title, c.version, c.section, c.page, c.content,
           1 - (c.embedding <=> CAST(:embedding AS vector)) AS similarity,
           (coalesce(1.0 / (:rrf_k + s.rank), 0) + coalesce(1.0 / (:rrf_k + k.rank), 0))::float8
               AS score
    FROM semantic s
    FULL JOIN keyword k USING (id)
    JOIN chunks c ON c.id = coalesce(s.id, k.id)
    JOIN documents d ON d.id = c.document_id
    ORDER BY score DESC, c.id
    LIMIT :limit
""")


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    document_id: UUID
    title: str
    version: int
    section: str | None
    page: int | None
    content: str
    similarity: float  # cosine similarity to the question, -1..1; used to decide relevance
    score: float  # fused rank score; only comparable within one search


def search(
    session: Session, embedder: Embedder, question: str, limit: int = 5, translation: str = ""
) -> list[Hit]:
    """Search for the question and, when given, its translation, fusing both rankings by rank.

    A translation reaches chunks in the other language: an English question about a page that
    exists only in Persian scores low against it, the same question in Persian does not.
    """
    queries = [normalize(q) for q in (question, translation) if q.strip()]
    embeddings = embedder.embed(queries)
    # An HNSW index scan finds its nearest rows before the version filter; iterative scanning
    # (pgvector 0.8) keeps searching until enough current chunks are found, in exact order.
    session.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))
    rankings = [
        [
            Hit(**row._mapping)
            for row in session.execute(
                _SEARCH,
                {
                    "embedding": str(embedding),
                    "terms": keyword_query(query),
                    "candidates": CANDIDATES,
                    "rrf_k": RRF_K,
                    "limit": CANDIDATES if len(queries) > 1 else limit,
                },
            )
        ]
        for query, embedding in zip(queries, embeddings, strict=True)
    ]
    if len(rankings) == 1:
        return rankings[0]
    return _fuse(rankings)[:limit]


def _fuse(rankings: list[list[Hit]]) -> list[Hit]:
    """Reciprocal Rank Fusion of several rankings; a chunk keeps its best similarity."""
    best: dict[int, Hit] = {}
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            if hit.chunk_id not in best or hit.similarity > best[hit.chunk_id].similarity:
                best[hit.chunk_id] = hit
    order = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [replace(best[chunk_id], score=scores[chunk_id]) for chunk_id in order]


def keyword_query(question: str) -> str:
    """An AND query of the question's meaningful words, e.g. `cdn & کش & پاک`.

    AND, not OR: an OR query matches nearly every chunk on common domain words and pushed the
    right chunk down (Recall@5 0.90 → 0.85 in the evaluation). With AND, keyword search adds a
    result only when every word appears, as with exact names and error codes.

    Words are split at anything that is not a letter or digit (including the half-space and `_`),
    the same way Postgres splits the indexed text.
    """
    words = re.findall(r"[^\W_]+", question.lower())
    return " & ".join(dict.fromkeys(w for w in words if w not in STOPWORDS))
