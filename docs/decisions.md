# Decision records

Short records of the choices that shape this project: the context, the decision, and what it costs.
All of them are accepted and implemented.

## 1. Python, as a single service

**Context.** The challenge leaves language and tools open and states that solution quality and the reasoning behind
technical decisions matter more than the number of tools or architectural complexity. The work that is specific to
this problem is AI and text work: structure-aware chunking, Persian text handling, embedding-model evaluation, and
optionally reranking.

**Decision.** Python 3.14 with FastAPI, as one service.

**Why.** Python is where the libraries for that work live and are maintained: text splitters, information-retrieval
evaluation, Persian NLP, and rerankers if retrieval needs one. Embeddings and the LLM are reached over HTTP, so any
language could call them; the difference is everything around those calls. FastAPI generates the OpenAPI spec and
Swagger UI from type hints, which covers a required deliverable with no extra code.

**Alternatives.** Go or TypeScript: equally good at HTTP, Postgres and concurrency, but would mean hand-writing
the text and evaluation pieces. A Go API plus a Python AI service: two deployables, a network contract between
them, and the document-version switch spread across two processes — complexity the problem does not need.

**Version.** 3.14 rather than 3.15 (stable since 2026-10-09): measured on 2026-10-28, the project's tests pass on
3.15, but `pyyaml` and `zstandard` publish no 3.15 wheels yet, so the Docker build would need a C compiler.
Upgrading is a one-line change once they do.

**Cost.** Python is slower than Go for CPU-bound work; none of this service's hot paths are CPU-bound (they wait
on the database, the embedding API and the LLM).

## 2. Postgres + pgvector as the only store

**Decision.** Documents, versions, chunks, vectors and the keyword index live in one Postgres database.

**Why.** The hardest requirement is that edits and deletes take effect atomically and outdated content is never used.
With everything in one database, switching a document to its new version is one ACID transaction, and every search
filters on the current version inside the same query. A separate vector database would need two systems to be
updated consistently.

**Cost.** pgvector's HNSW index supports up to 2,000 dimensions for `vector` and 4,000 for `halfvec`, so 3,072-dim
models need `halfvec`. At very large scale a dedicated vector engine may be faster; not a concern at this size.

**Indexes, verified with `EXPLAIN ANALYZE`.** Each search reads `chunks` directly (no shared CTE, which Postgres
would materialise and scan without indexes) and numbers ranks only after `LIMIT`, so neither sorts the whole
table. On the evaluation corpus (1,205 chunks with the demo files) the planner scans vectors exactly (cheaper at this size) and uses the GIN index for
keywords: 9 ms per search. With full scans priced out, as on a large table, it uses the HNSW index; pgvector 0.8's
`hnsw.iterative_scan = strict_order` keeps the index searching until enough chunks of current versions are found,
so the version filter cannot leave a question with too few results. Retrieval quality was identical on both paths
(Recall@5 0.897).

## 3. Build the RAG pipeline on small libraries, not a framework

**Decision.** Use the official `openai` SDK for embeddings and chat (its built-in retries cover rate limits and
timeouts), SQLAlchemy + `pgvector` for storage and search, and `langchain-text-splitters`' recursive splitter for
chunking. Do not use LangChain's vector stores, LangGraph or LlamaIndex.

**Why.** Each library replaces code we would otherwise write, without taking over the parts the challenge grades.
`langchain-text-splitters` (stable 1.x) gives recursive splitting with overlap at paragraph, line and sentence
boundaries; Markdown headings — which become the citation's section — are split by our own code (below). Framework vector stores own their own tables
and write path, so the one-transaction version switch (decision 2) would have to work around them; the pipeline is
linear (retrieve → check → generate → verify), so a workflow engine like LangGraph adds nothing.

Checked against the packages and source (2026-10-28):
- LangChain's `MarkdownHeaderTextSplitter` drops every character Python does not consider printable
  (`markdown.py`: `filter(str.isprintable, …)`), which includes the Persian half-space — `پاک‌سازی` became
  `پاکسازی`. Found by an end-to-end upload of a Persian file; heading splitting is now ~20 lines in `ingest.py`,
  with a regression test.
- `langchain-postgres` is at 0.0.19 and requires `pgvector<0.4`, conflicting with the current `pgvector` 0.5.
- LlamaIndex's `update_ref_doc` (`llama-index-core`, `indices/base.py`) deletes the document and then inserts it
  again, re-embedding in between. During that window the document is missing from search, and if the embedding
  call fails the document is gone — the opposite of the version switch this service needs.
- `llama-index-vector-stores-postgres` 0.9.0 requires `sqlalchemy<2.1` and `psycopg2`; this project uses
  SQLAlchemy 2.1 and psycopg 3.

## 4. LLM behind the OpenAI-compatible chat API

**Context.** The challenge provides embedding models only and leaves the rest of the architecture open.

**Decision.** Call the LLM through the OpenAI-compatible `chat/completions` API, configured by `LLM_BASE_URL`,
`LLM_API_KEY` and `LLM_MODEL`. Locally, `compose.yaml` runs **Ollama** with **Gemma 4 (`gemma4:e4b-it-qat`)**; the
first start downloads the model, later starts reuse it. Nothing besides Docker is needed, and no API key.

**Why Ollama.** It is the market standard for running models on one machine (checked 2026-10-28: 185k GitHub
stars, 193M Docker pulls), runs the same on Linux, macOS and Windows, and uses an NVIDIA GPU on Linux when given one.
Alternatives: **vLLM** (96k stars) is the standard for production serving on GPU servers and is what this service
would use at scale — the switch is configuration, because both speak the same API; **Docker Model Runner** (0.7k
stars) integrates with Compose but is new and needs Docker Desktop or a plugin; **LM Studio** is an excellent
desktop app (llama.cpp and MLX engines) but is closed source and has no official Docker image, so `make run` could
not include it — it still works with this service by pointing `LLM_BASE_URL` at it; Hugging Face TGI is archived;
a hosted API (ArvanCloud AIaaS, OpenAI) costs per token and would require reviewers to have a key.

**Why Gemma 4 e4b, QAT build.** The newest open model family that fits a laptop (~6 GB); QAT (quantization-aware
training) keeps more quality at 4-bit than quantizing afterwards. Measured through the API on 88 questions (decision 6):
all 20 unanswerable questions refused, 57 of 68 answerable ones answered with a citation to a labelled page.

**Thinking off.** Gemma 4 reasons before answering by default: ~690 hidden tokens for a 26-token answer. With
`reasoning_effort: none` (`LLM_REASONING_EFFORT`, set in `compose.yaml`) the same answer took 2.7 s instead of
13.3 s on an Apple GPU, and on the questions both runs covered, thinking produced more wrong answers (5 vs 3), not
fewer. Retrieval and the two grounding gates do the work that matters; the model only phrases the evidence.
The parameter is sent only when set, because models without reasoning reject it.

**Speed.** Docker on macOS cannot use the Apple GPU, so in `make run` on a Mac Gemma runs on the CPU: measured
2026-11-02 on the evaluation corpus with thinking off, about 50 s for the first answer (the model loads), then
5–11 s. With Ollama running natively (README) answers take 1–4 s; on Linux the Ollama container uses an NVIDIA GPU
when given one.
Larger options: `gemma4:12b` (7.7 GB); Qwen 3.6/3.8 start at ~16–18 GB.

**Grounding.** Two gates: if the best chunk is below the similarity threshold → "not enough information" without
calling the LLM; otherwise the LLM gets all top-5 chunks, because a weaker match can hold the answer (a manual's
reset steps on the page after its title). Filtering each chunk by the threshold had dropped such pages: answering
from all five raised the end-to-end result from 56/68 to 57/68, still with 20/20 refused (measured 2026-11-02); after the LLM, the answer must cite at least one source, only sources it was given, and every number it
states must appear in a source it cites (digits are normalised on both sides, so a Persian answer is checked
against an English source; list numbering is ignored). On the 68 answerable evaluation questions the number check
blocked none (56/68 answered and 20/20 refused, before the change above). It does not check that the wording
of a sentence is supported; that would need an NLI model or an LLM judge per sentence. Each citation quotes the
sentence of its chunk that shares the most words and numbers with the answer (the whole chunk when none does, as
for an answer in another language) and also returns the whole chunk. Source text is
HTML-escaped inside `<source>` tags so a document cannot close its tag and pose as instructions. The threshold is
a cosine similarity, not the fused search score: the fused score only orders results and always has a winner.
The threshold (0.58 for BGE-M3) is calibrated by the evaluation, see decision 6.

## 5. PDF text extraction with poppler (`pdftotext`)

**Decision.** Extract PDF text with poppler's `pdftotext`, then normalise it.

**Why (measured on the challenge PDF, page 3).** `pdftotext` kept correct word spacing; `pypdf` merged words
(`فرضکنیدطراحیو`); PDFium merged words the same way; `pdfplumber` and the pure-Go `ledongthuc/pdf` returned
reversed Persian text; PyMuPDF was close to poppler but is AGPL-licensed. Poppler leaves two defects: bidi control
characters, which normalisation removes, and the split لا ligature (`اطالعات`), which it cannot repair (decision 8).

## 6. Embedding model: `Bge-m3`, refusal threshold 0.58

**Status: Accepted** (measured 2026-10-28 with `eval/evaluate.py`).

**Method.** Corpus: 138 ArvanCloud documentation pages (98 Persian, 40 English: CDN, object storage, AIaaS,
accounts), 1,201 chunks. Questions: 88, written in both languages, some asked in one language about a page that
exists only in the other; 68 are answerable (labelled with the pages that answer them) and 20 are not, many of them
near misses on the same topics (e.g. a price the pages never state). A question counts as found when one of the
top 5 chunks — what the language model receives — comes from a labelled page.

| Model | Dims | Recall@1 | Recall@5 | MRR@10 | Answer/abstain acc. | s/request (64 texts) |
|---|---|---|---|---|---|---|
| **Bge-m3** | 1024 | 0.63 | 0.90 | 0.75 | 0.93 | **1.4** |
| Embedding-3-Small | 1536 | 0.57 | 0.87 | 0.69 | 0.90 | 4.0 |
| Embedding-3-Large | 3072 | 0.65 | 0.96 | 0.78 | 0.92 | 4.5 |
| Gemini-embedding-001 | 3072 | 0.76 | 0.97 | 0.86 | 0.95 | 55.8 |

**Decision.** `Bge-m3`. Gemini and Embedding-3-Large retrieve better, but:

- **Speed and reliability.** Every upload and every question waits for the embedding API. Gemini averaged 56 s per
  request because the gateway returned 504 timeouts and requests had to be retried; BGE-M3 was 3–40× faster.
- **Index size.** 3072 dimensions are 3× the storage and exceed pgvector's 2,000-dimension limit for an HNSW index
  on `vector`; they would need `halfvec`. 1024 fits directly.
- **Separation.** BGE-M3 separates answerable from unanswerable questions as well as the larger models (0.93), and
  that is what the "not enough information" gate depends on.
- **Open weights.** BGE-M3 (BAAI, MIT licence, multilingual including Persian) can be self-hosted, so documents never
  have to leave the company's infrastructure.

Switching is configuration plus re-embedding: `EMBEDDING_MODEL`, and for a 3072-dimension model a migration to
`halfvec(3072)`. If retrieval quality matters more than latency, Gemini is the upgrade path.

**Threshold.** Similarity is model-specific, so the threshold is calibrated per model. For BGE-M3 the weakest
answerable question's best chunk scored 0.588 and the strongest unanswerable near miss 0.618: no threshold separates
them perfectly. The most accurate threshold (what `eval/evaluate.py` reports, and the table's answer/abstain
accuracy) is 0.588 itself: 82 of 88 correct. The service rounds it down to `min_similarity = 0.58` for margin: it
refuses none of the 68 answerable questions and stops 13 of the 20 unanswerable ones before the language model is
called (81 of 88); the rest must be refused by the model (the second gate).

**Hybrid search, measured.** With the same embeddings, adding an OR keyword query lowered Recall@5 from 0.90 to
0.85: common domain words (CDN, آروان) matched almost every chunk and pushed the right one down. The keyword
query is therefore an AND of the question's words: on natural questions it changes nothing (0.90), and it still
adds exact matches such as error codes and product names, which embeddings handle poorly.

## 7. Build the service rather than deploy a RAG platform

**Context.** Mature open-source RAG products exist and are actively maintained (checked 2026-10-27): RAGFlow
(~93k GitHub stars), Dify (~160k), AnythingLLM (~68k), PrivateGPT (~58k), Onyx (~33k), Kotaemon (~26k).
Frameworks exist too: LangChain, LlamaIndex, Haystack, txtai.

**Decision.** For this challenge, build a small service, borrowing focused libraries (decision 3), instead of
deploying a platform. A platform such as RAGFlow would be a legitimate production choice for a team that needs a
ready document-chat product; the challenge asks for the system to be designed and implemented, so it is built here.

**Why.** The platforms are complete products — web UI, user management, connectors, several supporting services —
so adopting one would mean configuring someone else's system rather than designing one, and the parts the
challenge grades (versioned change management, retrieval quality, grounded citations, abstention, tests) would live
in code we neither wrote nor could fully explain. They remain useful as references for patterns such as hybrid
search and citation formats. In production, the build-or-adopt answer depends on the team: a platform wins when the
need is a generic chat-with-documents UI; a focused service wins when RAG is embedded in another product's API.

## 8. Persian normalisation without an NLP library

**Decision.** Normalise text with Python's `unicodedata` (NFKC) plus a small character map: Arabic ي/ى/ك → Persian
ی/ک, Persian and Arabic digits → ASCII, diacritics, tatweel and invisible direction marks removed, half-spaces kept,
and control characters other than line breaks and tabs removed (ArvanCloud's own pages contain NUL bytes, e.g.
`۲۵\x00۶` for `۲۵۶`, which Postgres cannot store).
The keyword index treats a half-space as a word break, so `می‌شود` matches `می شود`.

**Why (measured 2026-10-28).** `hazm`'s default normaliser damages technical text: `2.1` → `۲ ٫ ۱`,
`200,000` → `۲۰۰ , ۰۰۰`, and URLs are split apart. With those rules turned off, what remains is a character map —
about ten lines — while `hazm` would add `nltk` and `huggingface-hub` and pin Python below 3.14.

**Limitation.** PDFs that split the لا ligature (`اطلاعات` extracted as `اطالعات`) cannot be repaired without a
dictionary; vector search still matches these words by meaning, keyword search may miss them.

## 9. Tuning parameters

**Status: Accepted.** Every number the service depends on, where it is set and why.

| Parameter | Value | Where | Why |
|---|---|---|---|
| Chunk size / overlap | 1,200 / 150 characters | `ingest.py` | About a section of a documentation page, far below Bge-m3's 8,192-token input limit, so one chunk holds a complete answer and five fit easily in the prompt; the overlap (~12%) keeps a sentence cut at a boundary readable in both chunks. Chunks never cross a page or a Markdown section. Not tuned further: Recall@5 was 0.90 at this size. Constants, not settings: changing them means re-embedding everything. |
| Embedding batch | 64 chunks per request | `embeddings.py` | 64 × 1,200 = 76,800 characters, well under the provider's 200,000 per request. |
| Embedding rate limit | 120 requests/min (provider) | not throttled | The SDK retries 429, 5xx and timeouts with exponential backoff (`max_retries=5`, timeout 60 s). One upload of up to 7,680 chunks per minute fits; a larger one, or many concurrent questions, can exceed it and return 503. A production version would queue ingestion and throttle. |
| Vector column | `vector(1024)` | migration 0002, `models.py` | Bge-m3's dimension; another model needs a migration (3,072 dims: `halfvec`, decision 6). |
| HNSW index | pgvector defaults: m = 16, ef_construction = 64, ef_search = 40 | migration 0002 | Defaults are recall-oriented for this size; on 1,200 chunks the planner scans exactly anyway (decision 2). Tune `ef_search` when the table grows. |
| `hnsw.iterative_scan` | `strict_order` | `retrieval.py` | The version join filters after the index scan; iterative scanning keeps searching until enough current chunks are found, in exact distance order (decision 2). |
| Candidates per search | 20 | `retrieval.py` | Each of the two searches contributes up to 20 before fusion: 4× the final 5, enough for RRF to reorder. |
| RRF k | 60 | `retrieval.py` | The constant from the RRF paper (Cormack et al., 2009); it damps the weight of the top ranks so neither search dominates. Fusion is by rank, so cosine and `ts_rank_cd` never need a common scale. |
| top_k | 5 by default; `/search` ≤ 20, `/ask` ≤ 10 | `api.py` | Five chunks are what the evaluation measured (Recall@5) and what the model receives. |
| `min_similarity` | 0.58 | `config.py`, `MIN_SIMILARITY` | Decision 6. |
| LLM | `gemma4:e4b-it-qat`, temperature 0, `reasoning_effort` none | `config.py`, `compose.yaml` | Decision 4. Temperature 0: the same question gives the same answer, which makes answers reproducible and testable. |
| Timeouts / retries | embeddings 60 s / 5; LLM 120 s / 2; `pdftotext` 60 s | `embeddings.py`, `answering.py`, `ingest.py` | Embedding calls are short and rate-limited, so more retries; an LLM call is long, so fewer. A pathological PDF cannot hold a worker for more than a minute. |
| Upload limit | 20 MB | `config.py`, `MAX_UPLOAD_BYTES` | Uploads are processed synchronously; larger files belong in a background job. |
| Question length | 1–2,000 characters | `api.py` | A question, not a document. |
