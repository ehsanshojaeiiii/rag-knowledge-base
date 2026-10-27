# RAG Knowledge Base

A document question-answering service. Upload PDF, Markdown or text documents, ask questions in Persian or English,
and get answers grounded only in those documents — each with citations to the document, version and section it
came from. When the documents don't contain the answer, the service says so instead of guessing.

Documents can be added, edited and deleted at any time; answers never use deleted documents or outdated versions.

> Status: work in progress. The service skeleton, database and health check run; ingestion, retrieval and
> answering are being built. See [docs/architecture.md](docs/architecture.md) and the
> [decision records](docs/adr/).

## Quick start

Requirements: Docker with Compose.

```bash
cp .env.example .env        # then set EMBEDDING_API_KEY
docker compose up --build
```

- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- OpenAPI spec: http://localhost:8000/openapi.json

## Development

Requirements: [uv](https://docs.astral.sh/uv/) and Docker (tests start a throwaway Postgres container).

```bash
uv sync                     # install dependencies
docker compose up -d db     # start Postgres + pgvector
uv run alembic upgrade head # apply migrations
uv run fastapi dev          # run the API with reload
uv run pytest               # tests
uv run ruff check . && uv run ruff format --check . && uv run mypy src
```
