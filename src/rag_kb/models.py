from datetime import datetime
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, ForeignKey, func
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBEDDING_DIMENSIONS = 1024


class Base(DeclarativeBase):
    pass


class Document(Base):
    """A document; `version` points at the chunks that answers may use."""

    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(primary_key=True, server_default=func.uuidv7())
    title: Mapped[str]
    filename: Mapped[str]
    media_type: Mapped[str]
    content_hash: Mapped[str]
    version: Mapped[int] = mapped_column(server_default="1")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Chunk(Base):
    """A searchable piece of one version of a document."""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    version: Mapped[int]
    ordinal: Mapped[int]
    content: Mapped[str]
    section: Mapped[str | None]
    page: Mapped[int | None]
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
    search: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('simple', replace(content, U&'\\200C', ' '))", persisted=True),
    )
