from fastapi import FastAPI

from rag_kb import api

app = FastAPI(
    title="RAG Knowledge Base",
    version="0.1.0",
    description=(
        "Ask questions about uploaded documents and get answers grounded only in them, "
        "with citations to the document, version and section."
    ),
)
app.include_router(api.router)
