# RAG Knowledge Base

A document question-answering service. Upload PDF, Markdown or text documents, ask questions in Persian or English,
and get answers grounded only in those documents — each with citations to the document, version and section it
came from. When the documents don't contain the answer, the service says so instead of guessing.

Documents can be added, edited and deleted at any time; answers never use deleted documents or outdated versions.

> Status: work in progress. The service skeleton, database and health check run; ingestion, retrieval and
> answering are being built. See [docs/architecture.md](docs/architecture.md) and the
> [decision records](docs/adr/).

## Quick start

Requirements: Docker with Compose, `make`, and [Ollama](https://ollama.com) running on the host with the model
pulled (`ollama pull gemma4`) — or any OpenAI-compatible chat endpoint set in `.env` (`LLM_BASE_URL`, `LLM_API_KEY`,
`LLM_MODEL`).

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

## Tests

`make test` runs everything against a throwaway Postgres; the embedding API is replaced by a deterministic fake,
so tests are fast, free and repeatable. Model quality is measured separately (evaluation), because quality is a
score, not pass/fail.

| Requirement (from the challenge) | Covered by |
|---|---|
| Process text PDF, plain text and Markdown | `tests/test_ingest.py`, `TestDocumentProcessing` |
| Add, edit, delete; outdated content never used | `TestChangeManagement` — edit leaves only the new version; a failed edit keeps the old one; stale `If-Match` → 409; delete removes every chunk. `TestOutdatedContentIsNeverRetrieved` — search ignores old versions even if a stale chunk is left behind |
| Answer only from retrieved content, with traceable citations | `TestGroundedAnswers` — cites document, version and section; the model sees only retrieved sources; document text cannot break out of its source tag; answers without valid citations are withheld |
| Say clearly when information is missing | `TestNotEnoughInformation` — unrelated questions are refused without calling the model; model refusals are reported |
| Answers follow document changes | `TestAnswersFollowDocumentChanges` — after an edit only the new version reaches the model; after a delete the question is refused |
| Retrieve the parts related to the question | `TestRelevantRetrieval` — best chunk ranks first, exact identifiers, Persian half-space spellings, citation fields |
| API access with documented endpoints | `TestApiAccess` |
