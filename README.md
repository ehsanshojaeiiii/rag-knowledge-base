# RAG Knowledge Base

A document question-answering service. Upload PDF, Markdown or text documents, ask questions in Persian or English,
and get answers grounded only in those documents — each with citations to the document, version and section it
came from. When the documents don't contain the answer, the service says so instead of guessing.

Documents can be added, edited and deleted at any time; answers never use deleted documents or outdated versions.

## Quick start

Requirements: Docker with Compose, and `make`. The first `make run` downloads the language model
(`gemma4:e4b-it-qat`, ~6 GB) into an Ollama container; later starts reuse it.

```bash
cp .env.example .env        # then set EMBEDDING_API_KEY
make run                    # build and start Postgres, Ollama and the API (runs anywhere, CPU)
make run-gpu                # or: the model in Docker Model Runner, which uses the Apple GPU on a Mac
```

Docker containers on macOS cannot reach the Apple GPU; [Docker Model Runner](https://docs.docker.com/ai/model-runner/)
runs the model on the host's GPU while Compose still declares it ([compose.gpu.yaml](compose.gpu.yaml)). It needs
Docker Compose 2.38+ and Model Runner enabled (Docker Desktop: Settings → AI; Linux: the `docker-model-plugin`
package). Its first start downloads `ai/gemma4:e4b` (~6 GB). On Linux with an NVIDIA GPU, `make run` can instead give
the `ollama` service the GPU (Compose `deploy.resources.reservations.devices`).

If port 5432 is already used by another Postgres, set `DB_PORT=5433` in `.env`.

- Questions page: http://localhost:8000 — ask, read the answer with its sources, upload, replace and delete
  documents (one static page over the API; the brief needs no GUI, it makes a demo easier)
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
#                   "page": null, "quote": "<the supporting sentence>", "content": "<the whole chunk>"}]}
# or {"status": "insufficient_information", "answer": "The documents do not contain enough information …",
#     "related": [<up to three pages that came close: pointers to read, never an answer>]}

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
ask    → translate question (fa ↔ en) → hybrid search for both (vector + keyword, current versions only)
       → enough evidence?
       → LLM answers from numbered sources only → citations and numbers verified → answer + citations
```

Details and diagrams: [docs/architecture.md](docs/architecture.md). Every significant choice — language, storage,
libraries, models, PDF extraction, Persian handling — is recorded with its reasons and measurements in
[docs/decisions.md](docs/decisions.md). The key ones:

- **One Postgres database (with pgvector)** for documents, versions, vectors and the keyword index, so switching a
  document to a new version is a single transaction: questions see the old version or the new one, never a mix.
  The search query itself joins on the current version, so outdated chunks cannot be returned.
- **Hybrid search**: vector similarity finds paraphrases and cross-language matches; Postgres full-text search finds
  exact terms such as error codes; the two rankings are fused with Reciprocal Rank Fusion. Each chunk is embedded
  with its document title and section path, so it keeps the context it was cut from; the question is also searched
  translated into the other language (Persian ↔ English), so it reaches pages that exist in only one of them.
- **Two grounding gates**: below a similarity threshold the LLM is not called at all; after the LLM, an answer is
  returned only if it cites sources it was actually given, and every number it states (prices, limits) appears in
  a source it cites. Document text is escaped in the prompt, so a document
  cannot inject instructions. A refusal after similar pages were found lists up to three of them as `related`
  pointers — the reader gets a lead, and the service still states nothing it could not support.
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

Answer/abstain accuracy is at each model's most accurate threshold; for Bge-m3 that is 0.588, the similarity of the
weakest answerable question. The service rounds it down to 0.58, so the similarity gate refuses no answerable
question (accuracy 0.92: one more near miss reaches the model); the model and the citation check refuse the rest.

BGE-M3 is several times faster, fits a standard HNSW index and separates answerable from unanswerable questions as
well as the larger models; the reasoning and the upgrade path are in [decision 6](docs/decisions.md).

End to end through `/ask`, with Gemma 4 answering: **all 20 unanswerable questions were refused**, and **61 of the
68** answerable ones were answered with a citation to a labelled page (strict: a correct answer citing a related
overview page counts as wrong). Each improvement was kept only when this number rose
([decision 10](docs/decisions.md)):

| Change (measured 2026-11-02) | Answered (of 68) | Refused (of 20) |
|---|---|---|
| First version | 56 | 20 |
| All retrieved chunks reach the model once the best one clears the threshold | 57 | 20 |
| Each chunk embedded with its title and section (Recall@1 0.63 → 0.68) | 58 | 20 |
| The question also searched in the other language | **61** | **20** |

A second set, [eval/questions-holdout.jsonl](eval/questions-holdout.jsonl), was written after this tuning — more
English, English questions about Persian-only pages, casual spelling, near misses — to check that it generalises:
**19 of 22 answered, 4 of 4 refused.** Gemma 4's "thinking" is turned off (`LLM_REASONING_EFFORT=none`): on the same
questions it was 4–5× slower and not more accurate. Gemma 4 12B took 12–17 s per answer, about 3× e4b, on a laptop
with 24 GB.

```bash
uv run --group eval python eval/fetch_arvan_docs.py     # download the corpus (not committed)
uv run --group eval python eval/evaluate.py models      # compare the four embedding models
uv run --group eval python eval/evaluate.py system      # score the running service's search
uv run --group eval python eval/evaluate.py answers     # score the running service's answers, LLM included
EVAL_QUESTIONS=eval/questions-holdout.jsonl uv run --group eval python eval/evaluate.py answers   # held-out set
```

To try another ArvanCloud page, save it as Markdown in `data/pages/` and upload it from the questions page:

```bash
uv run --group eval python eval/fetch_arvan_docs.py https://docs.arvancloud.ir/en/vod/speech-to-text
```

## Tests

Two kinds:

- `make test` (and CI): every requirement against a throwaway Postgres with pgvector — real SQL, migrations and
  transactions — while the embedding API and the LLM are replaced by deterministic fakes, so the tests are fast,
  free and repeatable.
- `make test-live` ([tests/test_live.py](tests/test_live.py)): real scenarios against the running stack, nothing
  faked — real embeddings, real LLM, real HTTP. Upload → answer with section citation → edit → new answer, old one
  gone → stale edit 409 → delete → refused; a Persian question answered from an English document; a PDF answer citing
  its page; instructions inside a document ignored; an off-topic question refused; an unsupported file 415. The facts
  are invented ("Zelora"), so an answer can only come from the uploaded document. Needs the API key and
  `make run` or `make run-gpu` first; skipped otherwise. Takes about a minute.

| Requirement (from the challenge) | Covered by |
|---|---|
| Process text PDF, plain text and Markdown | `tests/test_ingest.py`, `TestDocumentProcessing` — including each chunk embedded with its document title and section |
| Add, edit, delete; outdated content never used | `TestChangeManagement` — edit leaves only the new version; a failed edit keeps the old one; stale `If-Match` → 409; an edit overtaken by another while embedding → 409; delete removes every chunk. `TestOutdatedContentIsNeverRetrieved` — search ignores old versions even if a stale chunk is left behind |
| Retrieve the parts related to the question | `TestRelevantRetrieval` — best chunk ranks first, exact identifiers, Persian half-space spellings, citation fields. `TestGroundedAnswers` — a question translated into the other language reaches a page that exists only in that language |
| Answer only from retrieved content, with traceable citations | `TestGroundedAnswers` — cites document, version and section; once one source clears the threshold the model sees all retrieved sources, and only those; document text cannot break out of its source tag; answers without valid citations are withheld; an answer stating a number its source does not contain is withheld; each citation quotes the supporting sentence, never a heading |
| Say clearly when information is missing | `TestNotEnoughInformation` — unrelated questions are refused without calling the model and without pointers; model refusals are reported, with related pages as pointers, never as an answer |
| Answers follow document changes | `TestAnswersFollowDocumentChanges` — after an edit only the new version reaches the model; after a delete the question is refused |
| API access with documented endpoints | `TestApiAccess`, `tests/test_openapi.py` (the committed spec matches the code), `tests/test_ui.py` (the page is served and never parses API text as HTML) |

## Development

Requirements: [uv](https://docs.astral.sh/uv/) and Docker.

```bash
make dev                    # local API with auto-reload; Postgres and Ollama in Docker
make test                   # tests (fast, with fakes for the embedding API and the LLM)
make test-live              # real scenarios against the running stack (after make run or make run-gpu)
make check                  # lint + type checks + tests (what CI runs)
make format                 # fix formatting
```

## Limitations

- Scanned PDFs (images without a text layer) are rejected; OCR is out of scope.
- Plain-text files have no sections, so their citations point to the document and the quoted sentence; PDFs cite
  the page, Markdown the heading path. Headings inside PDFs are not detected.
- The citation check verifies that each `[n]` is a source the model was given and that every number appears in a
  cited source; it does not verify that the wording of each sentence is supported (that needs an NLI or LLM judge).
- Embedding requests are not throttled to the provider's 120 requests per minute; the SDK retries after a 429. A
  very large upload, or many questions at once, can exceed the limit and get 503.
- Misses: of the 22 answerable held-out questions, 3 were refused. In two, a page on the same topic ranked above the
  exact page (account registration above the wallet page, general caching above the download bucket); in one the
  exact page was retrieved and the 4B model still judged it insufficient (account-level firewall rules) — that page
  is then the first `related` pointer. A cross-encoder reranker (e.g. bge-reranker-v2-m3) over the top 20 is the next
  step for the first kind, a larger model for the second; both were measured and left out on a 24 GB laptop.
- Answers take 1–2 s to refuse and 10–25 s for a long answer on a laptop GPU (Gemma 4 e4b, ~50 tokens/s); the
  translation adds one short model call. Streaming the answer would show text as it is written.
- A question in one language about a short document in the other scores lower similarity and can still be refused
  when the answer is there; the threshold favours refusing over guessing.
- PDFs that split the Persian لا ligature during text extraction keep that defect; vector search still matches
  such words by meaning, keyword search may miss them.
- Uploads are processed synchronously; very large documents would move to a background job in production.
- `POST /documents` always creates a new document, so uploading the same file twice stores it twice; new versions
  of a document go through `PUT`.
- No authentication: the API is meant to run behind a gateway that provides it.
