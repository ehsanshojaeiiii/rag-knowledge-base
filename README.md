# RAG Knowledge Base

A document question-answering service. Upload PDF, Markdown or text documents, ask questions in Persian or English,
and get answers grounded only in those documents — each with citations to the document, version and section it
came from. When the documents don't contain the answer, the service says so instead of guessing.

Documents can be added, edited and deleted at any time; answers never use deleted documents or outdated versions.

## Quick start

Requirements: Docker with Compose, and `make`. The first `make run` downloads the language model
(`gemma4:e4b-it-qat`, ~6 GB) into an Ollama container; later starts reuse it. On Linux with an NVIDIA GPU, give the
`ollama` service the GPU (Compose `deploy.resources.reservations.devices`) for faster answers. Docker on macOS
cannot use the Apple GPU; for GPU speed there, run Ollama natively and point the API at it with a git-ignored
`compose.override.yaml`:

```yaml
# OLLAMA_HOST=127.0.0.1:11435 ollama serve   (then: ollama pull gemma4:e4b-it-qat)
services:
  api:
    environment:
      LLM_BASE_URL: http://host.docker.internal:11435/v1
```

```bash
cp .env.example .env        # then set EMBEDDING_API_KEY
make run                    # build and start Postgres, Ollama and the API
```

- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- OpenAPI spec: http://localhost:8000/openapi.json, also committed as [docs/openapi.json](docs/openapi.json)
  (import it into Postman to get a ready collection)

Stop with `make down`.

## Using the API

```bash
# Upload a document (PDF with a text layer, Markdown or plain text)
curl -F file=@pricing.md http://localhost:8000/documents
# → {"id": "019a…", "title": "pricing", "version": 1, …}

# Ask a question
curl http://localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "قیمت پلن پایه چقدر است؟"}'
# → {"status": "answered", "answer": "… [1].",
#    "citations": [{"source": 1, "title": "pricing", "version": 1, "section": "Pricing > Basic plan",
#                   "page": null, "quote": "…"}]}
# or {"status": "insufficient_information", "answer": "The documents do not contain enough information …"}

# Replace a document with a new version; If-Match guards against overwriting someone else's edit (409)
curl -X PUT -H 'If-Match: 1' -F file=@pricing.md http://localhost:8000/documents/019a…

# Delete it — it stops being used immediately
curl -X DELETE http://localhost:8000/documents/019a…

# Inspect retrieval without generating an answer
curl http://localhost:8000/search -H 'Content-Type: application/json' -d '{"query": "basic plan price"}'
```

## How it works

```
upload → parse (poppler / Markdown headings / text) → normalise Persian → split into chunks → embed → Postgres
ask    → embed question → hybrid search (vector + keyword, current versions only) → enough evidence?
       → LLM answers from numbered sources only → every [n] citation verified → answer + citations
```

Details and diagrams: [docs/architecture.md](docs/architecture.md). Every significant choice — language, storage,
libraries, models, PDF extraction, Persian handling — is recorded with its reasons and measurements in
[docs/decisions.md](docs/decisions.md). The key ones:

- **One Postgres database (with pgvector)** for documents, versions, vectors and the keyword index, so switching a
  document to a new version is a single transaction: questions see the old version or the new one, never a mix.
  The search query itself joins on the current version, so outdated chunks cannot be returned.
- **Hybrid search**: vector similarity finds paraphrases and cross-language matches; Postgres full-text search finds
  exact terms such as error codes; the two rankings are fused with Reciprocal Rank Fusion.
- **Two grounding gates**: below a similarity threshold the LLM is not called at all; after the LLM, an answer is
  returned only if it cites sources it was actually given. Document text is escaped in the prompt, so a document
  cannot inject instructions.
- **Provider-agnostic models**: embeddings and the LLM are reached through the OpenAI-compatible API, so ArvanCloud
  AIaaS, vLLM or a hosted model replace the local Ollama by configuration only.

## Evaluation

`eval/` measures the service on ArvanCloud's public documentation: 138 pages in Persian and English, and 88
labelled questions in both languages, 20 of which the pages cannot answer (many are near misses on the same
topics). It chose the embedding model, calibrated the "not enough information" threshold and caught a weakness in
the first keyword search.

| Embedding model | Dims | Recall@1 | Recall@5 | MRR@10 | Answer/abstain acc. | s/request |
|---|---|---|---|---|---|---|
| **Bge-m3** (chosen) | 1024 | 0.63 | 0.90 | 0.75 | 0.93 | 1.4 |
| Embedding-3-Small | 1536 | 0.57 | 0.87 | 0.69 | 0.90 | 4.0 |
| Embedding-3-Large | 3072 | 0.65 | 0.96 | 0.78 | 0.92 | 4.5 |
| Gemini-embedding-001 | 3072 | 0.76 | 0.97 | 0.86 | 0.95 | 55.8 |

BGE-M3 is several times faster, fits a standard HNSW index and separates answerable from unanswerable questions as
well as the larger models; the reasoning and the upgrade path are in [decision 6](docs/decisions.md).

End to end through `/ask`, with Gemma 4 answering: **all 20 unanswerable questions were refused**, and 55 of the 68
answerable ones were answered with a citation to a labelled page (strict: a correct answer citing a related overview
page counts as wrong). Gemma 4's "thinking" is turned off (`LLM_REASONING_EFFORT=none`): on the same questions it was
4–5× slower and not more accurate.

```bash
uv run --group eval python eval/fetch_arvan_docs.py     # download the corpus (not committed)
uv run --group eval python eval/evaluate.py models      # compare the four embedding models
uv run --group eval python eval/evaluate.py system      # score the running service's search
uv run --group eval python eval/evaluate.py answers     # score the running service's answers, LLM included
```

## Tests

`make test` runs everything against a throwaway Postgres; the embedding API and the LLM are replaced by
deterministic fakes, so tests are fast, free and repeatable.

| Requirement (from the challenge) | Covered by |
|---|---|
| Process text PDF, plain text and Markdown | `tests/test_ingest.py`, `TestDocumentProcessing` |
| Add, edit, delete; outdated content never used | `TestChangeManagement` — edit leaves only the new version; a failed edit keeps the old one; stale `If-Match` → 409; delete removes every chunk. `TestOutdatedContentIsNeverRetrieved` — search ignores old versions even if a stale chunk is left behind |
| Retrieve the parts related to the question | `TestRelevantRetrieval` — best chunk ranks first, exact identifiers, Persian half-space spellings, citation fields |
| Answer only from retrieved content, with traceable citations | `TestGroundedAnswers` — cites document, version and section; the model sees only retrieved sources; document text cannot break out of its source tag; answers without valid citations are withheld |
| Say clearly when information is missing | `TestNotEnoughInformation` — unrelated questions are refused without calling the model; model refusals are reported |
| Answers follow document changes | `TestAnswersFollowDocumentChanges` — after an edit only the new version reaches the model; after a delete the question is refused |
| API access with documented endpoints | `TestApiAccess`, `tests/test_openapi.py` (the committed spec matches the code) |

## Development

Requirements: [uv](https://docs.astral.sh/uv/) and Docker.

```bash
make dev                    # local API with auto-reload; Postgres and Ollama in Docker
make test                   # tests
make check                  # lint + type checks + tests (what CI runs)
make format                 # fix formatting
```

## Limitations

- Scanned PDFs (images without a text layer) are rejected; OCR is out of scope.
- PDFs that split the Persian لا ligature during text extraction keep that defect; vector search still matches
  such words by meaning, keyword search may miss them.
- Uploads are processed synchronously; very large documents would move to a background job in production.
