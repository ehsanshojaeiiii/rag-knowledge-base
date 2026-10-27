from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from rag_kb.db import SessionDep

router = APIRouter(tags=["health"])


class Health(BaseModel):
    status: str
    pgvector: str


@router.get("/health")
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
