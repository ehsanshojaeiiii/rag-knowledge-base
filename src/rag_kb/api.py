from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from rag_kb import answering, documents, retrieval
from rag_kb.config import get_settings
from rag_kb.db import SessionDep
from rag_kb.embeddings import Embedder, get_embedder
from rag_kb.models import Document

router = APIRouter()

EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
LLMDep = Annotated[answering.LLM, Depends(answering.get_llm)]
FileField = Annotated[
    UploadFile, File(description="PDF (text, not scanned), Markdown or plain text")
]
TitleField = Annotated[str | None, Form(description="Defaults to the file name")]


class Health(BaseModel):
    status: str
    pgvector: str


class DocumentOut(BaseModel):
    id: UUID
    title: str
    filename: str
    media_type: str
    version: int
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, document: Document) -> DocumentOut:
        return cls.model_validate(document, from_attributes=True)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10, description="Chunks to consider as sources")


@router.get("/health", tags=["health"])
def health(session: SessionDep) -> Health:
    """Report whether the database is reachable and pgvector is installed."""
    try:
        version = session.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).scalar_one_or_none()
    except SQLAlchemyError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database unavailable") from exc
    if version is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "pgvector is not installed")
    return Health(status="ok", pgvector=version)


@router.post("/documents", status_code=status.HTTP_201_CREATED, tags=["documents"])
def create_document(
    session: SessionDep, embedder: EmbedderDep, file: FileField, title: TitleField = None
) -> DocumentOut:
    """Upload a document. It is searchable as soon as this returns."""
    data = _read(file)
    return DocumentOut.of(documents.create(session, embedder, _name(file), data, title))


@router.get("/documents", tags=["documents"])
def list_documents(session: SessionDep) -> list[DocumentOut]:
    return [DocumentOut.of(d) for d in documents.list_all(session)]


@router.get("/documents/{document_id}", tags=["documents"])
def get_document(session: SessionDep, document_id: UUID) -> DocumentOut:
    return DocumentOut.of(documents.get(session, document_id))


@router.put("/documents/{document_id}", tags=["documents"])
def replace_document(
    session: SessionDep,
    embedder: EmbedderDep,
    document_id: UUID,
    file: FileField,
    title: TitleField = None,
    if_match: Annotated[
        int | None,
        Header(description="Expected current version; a mismatch returns 409"),
    ] = None,
) -> DocumentOut:
    """Replace a document's content with a new version.

    Answers keep using the previous version until the new one is fully processed, then switch
    at once. Uploading identical content does not create a new version.
    """
    data = _read(file)
    return DocumentOut.of(
        documents.replace(session, embedder, document_id, _name(file), data, title, if_match)
    )


@router.delete(
    "/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["documents"]
)
def delete_document(session: SessionDep, document_id: UUID) -> None:
    """Delete a document. It is excluded from every answer from this moment on."""
    documents.remove(session, document_id)


@router.post("/search", tags=["search"])
def search(
    session: SessionDep, embedder: EmbedderDep, request: SearchRequest
) -> list[retrieval.Hit]:
    """Return the chunks most relevant to a query, without generating an answer.

    Only the current version of existing documents is searched.
    """
    return retrieval.search(session, embedder, request.query, request.top_k)


@router.post("/ask", tags=["answers"])
def ask(
    session: SessionDep, embedder: EmbedderDep, llm: LLMDep, request: AskRequest
) -> answering.Answer:
    """Answer a question from the documents, with a citation for every source used.

    `status` is `insufficient_information` when the documents do not support an answer.
    """
    return answering.answer(session, embedder, llm, request.question, request.top_k)


def _read(file: UploadFile) -> bytes:
    limit = get_settings().max_upload_bytes
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"File is larger than {limit} bytes")
    return data


def _name(file: UploadFile) -> str:
    if not file.filename:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "The upload has no file name")
    return file.filename
