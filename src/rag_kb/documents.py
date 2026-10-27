"""Add, replace and delete documents so that answers only ever see one complete, current version.

Parsing and embedding are slow and happen outside any database transaction; the switch to a new
version is one short transaction, so readers see either the old version or the new one.
"""

import hashlib
from pathlib import PurePath
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from rag_kb.embeddings import Embedder
from rag_kb.ingest import TextChunk, chunk, media_type
from rag_kb.models import Chunk, Document


class DocumentNotFound(LookupError):
    pass


class VersionConflict(RuntimeError):
    """The document changed since the caller read it."""


def create(
    session: Session, embedder: Embedder, filename: str, data: bytes, title: str | None
) -> Document:
    kind = media_type(filename)
    chunks = _embed(embedder, chunk(data, kind))
    document = Document(
        title=title or PurePath(filename).stem,
        filename=filename,
        media_type=kind,
        content_hash=_hash(data),
        version=1,
    )
    session.add(document)
    session.flush()
    session.add_all(_rows(document.id, 1, chunks))
    session.commit()
    return document


def replace(
    session: Session,
    embedder: Embedder,
    document_id: UUID,
    filename: str,
    data: bytes,
    title: str | None,
    expected_version: int | None = None,
) -> Document:
    current = get(session, document_id)
    if expected_version is not None and expected_version != current.version:
        raise VersionConflict(f"Document is at version {current.version}, not {expected_version}")
    if _hash(data) == current.content_hash and filename == current.filename:
        if title and title != current.title:
            current.title = title
            session.commit()
        return current

    read_version = current.version
    session.rollback()  # end the read transaction before the slow part
    kind = media_type(filename)
    chunks = _embed(embedder, chunk(data, kind))

    new_version = read_version + 1
    values: dict[str, object] = {
        "version": new_version,
        "filename": filename,
        "media_type": kind,
        "content_hash": _hash(data),
        "updated_at": func.now(),
    }
    if title:
        values["title"] = title
    switched = session.execute(
        update(Document)
        .where(Document.id == document_id, Document.version == read_version)
        .values(values)
        .returning(Document)
    ).scalar_one_or_none()
    if switched is None:
        session.rollback()
        raise VersionConflict(
            "Document was changed or deleted while this version was being prepared"
        )
    session.add_all(_rows(document_id, new_version, chunks))
    session.execute(
        delete(Chunk).where(Chunk.document_id == document_id, Chunk.version != new_version)
    )
    session.commit()
    return switched


def remove(session: Session, document_id: UUID) -> None:
    deleted = session.execute(
        delete(Document).where(Document.id == document_id).returning(Document.id)
    ).scalar_one_or_none()
    if deleted is None:
        raise DocumentNotFound(str(document_id))
    session.commit()


def get(session: Session, document_id: UUID) -> Document:
    document = session.get(Document, document_id)
    if document is None:
        raise DocumentNotFound(str(document_id))
    return document


def list_all(session: Session) -> list[Document]:
    return list(session.scalars(select(Document).order_by(Document.created_at.desc())))


def _embed(embedder: Embedder, chunks: list[TextChunk]) -> list[tuple[TextChunk, list[float]]]:
    return list(zip(chunks, embedder.embed([c.content for c in chunks]), strict=True))


def _rows(
    document_id: UUID, version: int, chunks: list[tuple[TextChunk, list[float]]]
) -> list[Chunk]:
    return [
        Chunk(
            document_id=document_id,
            version=version,
            ordinal=ordinal,
            content=text.content,
            section=text.section,
            page=text.page,
            embedding=vector,
        )
        for ordinal, (text, vector) in enumerate(chunks)
    ]


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
