# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository.

## What this is

A document Q&A service using retrieval-augmented generation: PDF/Markdown/text in, grounded answers with citations
out, with versioned document updates. Read [docs/architecture.md](docs/architecture.md) and
[docs/decisions.md](docs/decisions.md) before changing behaviour.

## Commands

```bash
uv sync                                   # install
docker compose up -d db                   # Postgres + pgvector for local runs
uv run alembic upgrade head               # migrate
uv run fastapi dev                        # run with reload (entrypoint in pyproject.toml)
uv run pytest                             # tests (start a throwaway Postgres via testcontainers)
uv run ruff check . && uv run ruff format --check . && uv run mypy src tests
docker compose up --build                 # full stack
```

## Invariants — never break these

- An answer may only use chunks of the **current version** of **non-deleted** documents. Enforce it in the search
  query itself, not by filtering results afterwards.
- A document version switch (and a delete) is **one database transaction**.
- Every citation in an answer must refer to a chunk that was actually retrieved for that question; otherwise the
  answer is "not enough information".
- Secrets (`EMBEDDING_API_KEY`, `LLM_API_KEY`) come from the environment only — never in code, logs, tests or commits.

## Conventions

- Flat package `src/rag_kb/`: one module per responsibility. Add a module only when a new responsibility appears.
- FastAPI: follow the `fastapi` skill — `Annotated` parameters, return types as response models, plain `def`
  endpoints unless everything inside is async.
- Database: SQLAlchemy 2 typed ORM (not SQLModel — pre-1.0, and API schemas stay separate from tables);
  schema changes only through Alembic migrations in `migrations/versions/`.
- No LangChain vector stores, LangGraph or LlamaIndex (see decision 3). `langchain-text-splitters` is allowed for chunking.
- Type-check with mypy `--strict` (ty is still beta). Format and lint with Ruff.
- Tests: behaviour over implementation; integration tests use a real Postgres, never mocks of the database.
- Latest stable versions of dependencies; check the registry, never pin from memory.

## Skills

Project skills live in `.agents/skills/` (symlinked into `.claude/skills/`), pinned in `skills-lock.json`:
`fastapi`, `rag-implementation`, `llm-evaluation`, `python-testing-patterns`, `supabase-postgres-best-practices`.
Restore them with `npx skills experimental_install`.
