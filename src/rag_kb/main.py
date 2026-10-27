from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from rag_kb import api
from rag_kb.documents import DocumentNotFound, VersionConflict
from rag_kb.embeddings import EmbeddingUnavailable
from rag_kb.ingest import UnreadableDocument, UnsupportedDocument

app = FastAPI(
    title="RAG Knowledge Base",
    version="0.1.0",
    description=(
        "Ask questions about uploaded documents and get answers grounded only in them, "
        "with citations to the document, version and section."
    ),
)
app.include_router(api.router)

_ERROR_STATUS = {
    DocumentNotFound: status.HTTP_404_NOT_FOUND,
    VersionConflict: status.HTTP_409_CONFLICT,
    UnsupportedDocument: status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    UnreadableDocument: status.HTTP_422_UNPROCESSABLE_CONTENT,
    EmbeddingUnavailable: status.HTTP_503_SERVICE_UNAVAILABLE,
}


def _error(request: Request, exc: Exception) -> JSONResponse:
    message = "Document not found" if isinstance(exc, DocumentNotFound) else str(exc)
    return JSONResponse({"detail": message}, status_code=_ERROR_STATUS[type(exc)])


for _exception in _ERROR_STATUS:
    app.add_exception_handler(_exception, _error)
