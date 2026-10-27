# Decision records

Short records of the choices that shape this project: the context, the decision, and what it costs.
Status is **Accepted** unless marked **Proposed** (to be settled by measurement).

## 1. Python, as a single service

**Context.** The challenge leaves language and tools open and states that solution quality and the reasoning behind
technical decisions matter more than the number of tools or architectural complexity. The work that is specific to
this problem is AI and text work: token-aware chunking, Persian text handling, embedding-model evaluation, and
optionally reranking.

**Decision.** Python 3.13 with FastAPI, as one service.

**Why.** Python is where the libraries for that work live and are maintained: Hugging Face `tokenizers` (exact
token counts for BGE-M3), Persian NLP (`hazm`), information-retrieval metrics, and rerankers if retrieval needs one.
Embeddings and the LLM are reached over HTTP, so any language could call them; the difference is everything around
those calls. FastAPI generates the OpenAPI spec and Swagger UI from type hints, which covers a required deliverable
with no extra code.

**Alternatives.** Go or TypeScript: equally good at HTTP, Postgres and concurrency, but would mean hand-writing
the text and evaluation pieces. A Go API plus a Python AI service: two deployables, a network contract between
them, and the document-version switch spread across two processes — complexity the problem does not need.
Python 3.14: `hazm` does not support it yet.

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

## 3. Build the RAG pipeline on small libraries, not a framework

**Decision.** Use the official `openai` SDK for embeddings and chat, SQLAlchemy + `pgvector` for storage and search,
and `langchain-text-splitters` for chunking. Do not use LangChain's vector stores, LangGraph or LlamaIndex.

**Why.** Each library replaces code we would otherwise write, without taking over the parts the challenge grades.
`langchain-text-splitters` (stable 1.x) gives structure-aware Markdown splitting with heading metadata — which
becomes the citation's section — and recursive splitting with overlap. Framework vector stores own their own tables
and write path, so the one-transaction version switch (decision 2) would have to work around them; the pipeline is
linear (retrieve → check → generate → verify), so a workflow engine like LangGraph adds nothing. Measured:
`langchain-postgres` is at 0.0.19 and requires `pgvector<0.4`, conflicting with the current `pgvector` 0.5.

## 4. LLM behind the OpenAI-compatible chat API

**Context.** The challenge provides embedding models only and leaves the rest of the architecture open.

**Decision.** Call the LLM through the OpenAI-compatible `chat/completions` API, configured by `LLM_BASE_URL`,
`LLM_API_KEY` and `LLM_MODEL`. The default is a local Ollama model so the project runs with no extra credentials.

**Why.** It is the de-facto standard interface: ArvanCloud AIaaS, Ollama, vLLM and OpenAI all serve it, so switching
provider is configuration, not code. The specific model is chosen by measuring answer quality on Persian and
English questions (**Proposed** until measured).

## 5. PDF text extraction with poppler (`pdftotext`)

**Decision.** Extract PDF text with poppler's `pdftotext`, then normalise it.

**Why (measured on the challenge PDF, page 3).** `pdftotext` kept correct word spacing; `pypdf` merged words
(`فرضکنیدطراحیو`); PDFium merged words the same way; `pdfplumber` and the pure-Go `ledongthuc/pdf` returned
reversed Persian text; PyMuPDF was close to poppler but is AGPL-licensed. Poppler's two defects — bidi control
characters and the split لا ligature (`اطالعات`) — are handled by normalisation.

## 6. Embedding model

**Status: Proposed.** Candidates: `Bge-m3` (1024 d), `Embedding-3-Small` (1536 d), `Embedding-3-Large` (3072 d),
`Gemini-embedding-001` (3072 d). The choice will be made from Recall@k and MRR on a labelled Persian + English
question set, plus latency and index size. A first probe favoured `Bge-m3` (best separation of related vs unrelated
text, lowest latency); that is a hypothesis until the evaluation runs.

## 7. Build the service rather than deploy a RAG platform

**Context.** Mature open-source RAG products exist and are actively maintained (checked 2026-10-27): RAGFlow
(~93k GitHub stars), Dify (~160k), AnythingLLM (~68k), PrivateGPT (~58k), Onyx (~33k), Kotaemon (~26k).
Frameworks exist too: LangChain, LlamaIndex, Haystack, txtai.

**Decision.** Build a small service, borrowing focused libraries (decision 3), instead of deploying a platform.

**Why.** The platforms are complete products — web UI, user management, connectors, several supporting services —
so adopting one would mean configuring someone else's system rather than designing one, and the parts the
challenge grades (versioned change management, retrieval quality, grounded citations, abstention, tests) would live
in code we neither wrote nor could fully explain. They remain useful as references for patterns such as hybrid
search and citation formats. In production, the build-or-adopt answer depends on the team: a platform wins when the
need is a generic chat-with-documents UI; a focused service wins when RAG is embedded in another product's API.
