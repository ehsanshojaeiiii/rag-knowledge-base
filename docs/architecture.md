# Architecture

One FastAPI service and one Postgres database (with pgvector). Embeddings come from an OpenAI-compatible
embeddings API; answers are written by an LLM behind an OpenAI-compatible chat API.

## Flow

```mermaid
flowchart LR
    subgraph Ingest["Ingest — add or edit a document"]
        direction LR
        F[PDF / Markdown / Text] --> P[Parse] --> N[Normalise] --> C[Chunk] --> E[Embed]
    end
    subgraph Ask["Ask — answer a question"]
        direction LR
        Q[Question] --> QE[Embed] --> S[Hybrid search] --> G{Enough<br/>evidence?}
        G -- no --> NI[Not enough information]
        G -- yes --> L[LLM writes answer<br/>with citations] --> V[Verify citations] --> A[Answer + sources]
    end
    E --> DB[(Postgres + pgvector<br/>documents · versions · chunks)]
    DB --> S
    E -. embeddings API .- EMB[[Embedding model]]
    QE -. embeddings API .- EMB
    L -. chat API .- LLM[[LLM]]
```

| Step | What it does |
|---|---|
| Parse | Extract text; keep page numbers (PDF) and headings (Markdown). |
| Normalise | One spelling per character (Arabic ي/ك → Persian ی/ک), half-spaces, digits, strip PDF control characters. |
| Chunk | Split into overlapping pieces; each keeps document, version, section, page and character offsets. |
| Embed | Turn each chunk into a vector with the chosen embedding model. |
| Hybrid search | Vector similarity + keyword match, fused by rank, over **current versions of live documents only**. |
| Evidence check | If the best match is too weak, answer "not enough information" without calling the LLM. |
| Generate | The LLM answers only from numbered sources and cites them as `[n]`. |
| Verify | Every `[n]` must be a source we sent; an answer with no valid citation becomes "not enough information". |

## Editing a document without serving stale or mixed content

```mermaid
sequenceDiagram
    participant C as Client
    participant API
    participant DB as Postgres
    C->>API: PUT /documents/{id} (new file)
    API->>DB: create version n+1 (status: processing)
    Note over API,DB: parse, chunk and embed version n+1 —<br/>queries keep using version n meanwhile
    API->>DB: BEGIN — mark n+1 ready, set current_version = n+1 — COMMIT
    Note over DB: one transaction: readers see version n<br/>or version n+1, never a mix
    API-->>C: 200 OK (version n+1)
```

Search always filters `document is not deleted AND chunk.version = document.current_version`, so outdated chunks
can never be returned even before they are cleaned up. Deleting a document is a single update that excludes it from
every later search.
