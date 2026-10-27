# RAG Knowledge Base

A document question-answering service. Upload PDF, Markdown or text documents, ask questions in Persian or English,
and get answers grounded only in those documents — each with citations to the document, version and section it
came from. When the documents don't contain the answer, the service says so instead of guessing.

Documents can be added, edited and deleted at any time; answers never use deleted documents or outdated versions.

> Status: work in progress. The service skeleton, database and health check run; ingestion, retrieval and
> answering are being built. See [docs/architecture.md](docs/architecture.md) and the
> [decision records](docs/adr/).

## Quick start

Requirements: Docker with Compose, and `make`.

```bash
cp .env.example .env        # then set EMBEDDING_API_KEY
make run                    # build and start Postgres + API
```

- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- OpenAPI spec: http://localhost:8000/openapi.json

Stop with `make down`.

## Development

Requirements: [uv](https://docs.astral.sh/uv/) and Docker (tests start a throwaway Postgres container).

```bash
make dev                    # local API with auto-reload, Postgres in Docker
make test                   # tests
make check                  # lint + type checks + tests (what CI runs)
make format                 # fix formatting
```
