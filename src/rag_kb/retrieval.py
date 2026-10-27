"""Find the chunks most relevant to a question, from current versions of existing documents only.

Two searches run in one query and are fused by rank (Reciprocal Rank Fusion):
- semantic: nearest embeddings (cosine), good at paraphrases and cross-language questions;
- keyword: Postgres full-text match, good at exact terms such as product names and error codes.
"""

import re
from dataclasses import dataclass
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

_SEARCH = text("""
    WITH current_chunks AS (
        SELECT c.id, c.embedding, c.search
        FROM chunks c
        JOIN documents d ON d.id = c.document_id AND d.version = c.version
    ),
    semantic AS (
        SELECT id, row_number() OVER (ORDER BY embedding <=> CAST(:embedding AS vector)) AS rank
        FROM current_chunks
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :candidates
    ),
    keyword AS (
        SELECT id, row_number() OVER (ORDER BY ts_rank_cd(search, query) DESC) AS rank
        FROM current_chunks, to_tsquery('simple', :terms) AS query
        WHERE :terms <> '' AND search @@ query
        ORDER BY ts_rank_cd(search, query) DESC
        LIMIT :candidates
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


def search(session: Session, embedder: Embedder, question: str, limit: int = 5) -> list[Hit]:
    question = normalize(question)
    [embedding] = embedder.embed([question])
    rows = session.execute(
        _SEARCH,
        {
            "embedding": str(embedding),
            "terms": keyword_query(question),
            "candidates": CANDIDATES,
            "rrf_k": RRF_K,
            "limit": limit,
        },
    )
    return [Hit(**row._mapping) for row in rows]


def keyword_query(question: str) -> str:
    """An OR query of the question's meaningful words, e.g. `cdn | کش | پاک`.

    Words are split at anything that is not a letter or digit (including the half-space and `_`),
    the same way Postgres splits the indexed text.
    """
    words = re.findall(r"[^\W_]+", question.lower())
    return " | ".join(dict.fromkeys(w for w in words if w not in STOPWORDS))
