from datetime import datetime
from typing import Annotated, Any
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


class ErrorOut(BaseModel):
    detail: str


_ERRORS = {
    404: "No document with this id",
    409: "If-Match is not the current version, or another edit landed while this one was processed",
    413: "The file is larger than MAX_UPLOAD_BYTES (20 MB by default)",
    415: "Not a PDF, Markdown (.md) or text (.txt) file",
    503: "The embedding service, the language model or the database is unreachable",
}


def errors(*codes: int) -> dict[int | str, dict[str, Any]]:
    """OpenAPI entries for the domain errors an endpoint can return (main.py maps them)."""
    return {code: {"model": ErrorOut, "description": _ERRORS[code]} for code in codes}


_ANSWER_EXAMPLE = {
    "status": "answered",
    "answer": "The basic plan costs 12 dollars per month [1].",
    "citations": [
        {
            "source": 1,
            "document_id": "019a2b3c-4d5e-7f60-8a1b-2c3d4e5f6a7b",
            "title": "pricing",
            "version": 2,
            "section": "Pricing > Basic plan",
            "page": None,
            "chunk_id": 42,
            "quote": "The basic plan costs 12 dollars per month.",
            "content": "The basic plan costs 12 dollars per month. It includes 50 GB.",
        }
    ],
}


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
    query: str = Field(min_length=1, max_length=2000, examples=["پاک کردن کش CDN"])
    top_k: int = Field(default=5, ge=1, le=20)


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1, max_length=2000, examples=["How much does the basic plan cost?"]
    )
    top_k: int = Field(default=5, ge=1, le=10, description="Chunks to consider as sources")


@router.get("/health", tags=["health"], responses=errors(503))
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


@router.post(
    "/documents",
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
    responses=errors(413, 415, 503),
)
def create_document(
    session: SessionDep, embedder: EmbedderDep, file: FileField, title: TitleField = None
) -> DocumentOut:
    """Upload a document. It is searchable as soon as this returns.

    422 also means no text could be extracted (an empty file, or a scanned PDF without a text
    layer) or a text file that is not UTF-8.
    """
    data = _read(file)
    return DocumentOut.of(documents.create(session, embedder, _name(file), data, title))


@router.get("/documents", tags=["documents"])
def list_documents(session: SessionDep) -> list[DocumentOut]:
    """List all documents, newest first, each at its current version."""
    return [DocumentOut.of(d) for d in documents.list_all(session)]


@router.get("/documents/{document_id}", tags=["documents"], responses=errors(404))
def get_document(session: SessionDep, document_id: UUID) -> DocumentOut:
    """Return one document and its current version."""
    return DocumentOut.of(documents.get(session, document_id))


@router.put(
    "/documents/{document_id}", tags=["documents"], responses=errors(404, 409, 413, 415, 503)
)
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
    at once. Uploading identical content does not create a new version. 422 as for upload.
    """
    data = _read(file)
    return DocumentOut.of(
        documents.replace(session, embedder, document_id, _name(file), data, title, if_match)
    )


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["documents"],
    responses=errors(404),
)
def delete_document(session: SessionDep, document_id: UUID) -> None:
    """Delete a document. It is excluded from every answer from this moment on."""
    documents.remove(session, document_id)


@router.post("/search", tags=["search"], responses=errors(503))
def search(
    session: SessionDep, embedder: EmbedderDep, request: SearchRequest
) -> list[retrieval.Hit]:
    """Return the chunks most relevant to a query, without generating an answer.

    Only the current version of existing documents is searched. The query is searched as
    given; `/ask` also searches its translation into the other document language.
    """
    return retrieval.search(session, embedder, request.query, request.top_k)


@router.post(
    "/ask",
    tags=["answers"],
    responses={
        200: {"content": {"application/json": {"example": _ANSWER_EXAMPLE}}},
        **errors(503),
    },
)
def ask(
    session: SessionDep, embedder: EmbedderDep, llm: LLMDep, request: AskRequest
) -> answering.Answer:
    """Answer a question from the documents, with a citation for every source used.

    `status` is `insufficient_information` when the documents do not support an answer. Each
    citation's `quote` is the sentence of the source that best matches the answer; `content` is
    the whole retrieved chunk.
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
