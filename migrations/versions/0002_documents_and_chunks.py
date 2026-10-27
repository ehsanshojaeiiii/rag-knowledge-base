"""documents and chunks

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-28 00:02:52.584157

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE documents (
            id uuid PRIMARY KEY DEFAULT uuidv7(),
            title text NOT NULL,
            filename text NOT NULL,
            media_type text NOT NULL,
            content_hash text NOT NULL,
            version integer NOT NULL DEFAULT 1,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        )
    """)
    # Each chunk belongs to one version of a document. Search joins on
    # documents.version, so chunks of an older version are never returned.
    op.execute("""
        CREATE TABLE chunks (
            id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            document_id uuid NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
            version integer NOT NULL,
            ordinal integer NOT NULL,
            content text NOT NULL,
            section text,
            page integer,
            embedding vector(1024) NOT NULL,
            -- A half-space (U+200C) counts as a word break, so می‌شود matches می شود.
            search tsvector GENERATED ALWAYS AS (
                to_tsvector('simple', replace(content, U&'\\200C', ' '))
            ) STORED,
            UNIQUE (document_id, version, ordinal)
        )
    """)
    op.execute(
        "CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute("CREATE INDEX chunks_search_idx ON chunks USING gin (search)")


def downgrade() -> None:
    op.execute("DROP TABLE chunks")
    op.execute("DROP TABLE documents")
