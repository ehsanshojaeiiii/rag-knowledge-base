"""Measure retrieval quality on the Arvan docs question set.

    uv run --group eval python eval/evaluate.py models   # four embedding models, vector search
    uv run --group eval python eval/evaluate.py system   # the running API (hybrid search)
    uv run --group eval python eval/evaluate.py answers  # the running API, end to end with the LLM

A question is answered correctly when one of the top 5 chunks comes from a page labelled as
answering it — the same 5 chunks the answer step gives the language model. The Persian and English
versions of a page count as the same page. Unanswerable questions have no pages and are used to
calibrate the similarity threshold below which the service says there is not enough information.
"""

import json
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rag_kb.config import get_settings
from rag_kb.embeddings import OpenAICompatibleEmbedder
from rag_kb.ingest import chunk

CORPUS = Path("data/arvan-docs")
CACHE = Path("data/eval-cache")
QUESTIONS = Path("eval/questions.jsonl")
MODELS = ["Bge-m3", "Embedding-3-Small", "Embedding-3-Large", "Gemini-embedding-001"]
TOP_K = 5
API = "http://localhost:8000"


@dataclass
class Question:
    text: str
    pages: list[str]  # empty when the documents do not answer it


@dataclass
class Ranking:
    pages: list[str]  # page of each retrieved chunk, best first
    similarity: float  # cosine similarity of the best chunk


def page_of(filename: str) -> str:
    """fa__cdn__caching.md and en__cdn__caching.md are both page cdn/caching."""
    return "/".join(Path(filename).stem.split("__")[1:])


def load_questions() -> list[Question]:
    lines = QUESTIONS.read_text(encoding="utf-8").splitlines()
    return [Question(item["q"], item["pages"]) for item in map(json.loads, lines) if item]


def load_chunks() -> tuple[list[str], list[str]]:
    texts, pages = [], []
    for path in sorted(CORPUS.glob("*.md")):
        for piece in chunk(path.read_bytes(), "text/markdown"):
            texts.append(piece.content)
            pages.append(page_of(path.name))
    return texts, pages


def embed(model: str, texts: list[str], name: str) -> tuple[np.ndarray, float]:
    """Unit-length embeddings, cached on disk; also the seconds per request when not cached."""
    path = CACHE / f"{model}-{name}-{len(texts)}.npy"
    if path.exists():
        return np.load(path), float("nan")
    settings = get_settings()
    assert settings.embedding_api_key, "EMBEDDING_API_KEY is not set"
    embedder = OpenAICompatibleEmbedder(
        settings.embedding_base_url, settings.embedding_api_key.get_secret_value(), model
    )
    started = time.perf_counter()
    vectors = np.array(embedder.embed(texts), dtype=np.float32)
    seconds_per_request = (time.perf_counter() - started) / -(-len(texts) // 64)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors, seconds_per_request


def score(questions: list[Question], rankings: list[Ranking]) -> dict[str, float]:
    answerable = [(q, r) for q, r in zip(questions, rankings, strict=True) if q.pages]
    ranks = [_first_relevant_rank(q, r) for q, r in answerable]
    found = [r for r in ranks if r is not None and r <= TOP_K]
    threshold, accuracy = calibrate(questions, rankings)
    return {
        "recall@1": sum(r == 1 for r in ranks) / len(ranks),
        f"recall@{TOP_K}": len(found) / len(ranks),
        "mrr@10": sum(1 / r for r in ranks if r) / len(ranks),
        "threshold": threshold,
        "abstain_accuracy": accuracy,
    }


def _first_relevant_rank(question: Question, ranking: Ranking) -> int | None:
    return next((i for i, p in enumerate(ranking.pages[:10], 1) if p in question.pages), None)


def calibrate(questions: list[Question], rankings: list[Ranking]) -> tuple[float, float]:
    """The similarity threshold that best separates answerable from unanswerable questions."""
    labelled = [(r.similarity, bool(q.pages)) for q, r in zip(questions, rankings, strict=True)]
    candidates = sorted({s for s, _ in labelled})
    best = max(candidates, key=lambda t: sum((s >= t) == a for s, a in labelled))
    return best, sum((s >= best) == a for s, a in labelled) / len(labelled)


def evaluate_models() -> None:
    questions, (texts, pages) = load_questions(), load_chunks()
    print(f"{len(questions)} questions, {len(texts)} chunks from {len(set(pages))} pages\n")
    print(
        "| Model | Dims | Recall@1 | Recall@5 | MRR@10 | Threshold | Answer/abstain acc. | s/req |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for model in MODELS:
        chunk_vectors, latency = embed(model, texts, "chunks")
        question_vectors, _ = embed(model, [q.text for q in questions], "questions")
        similarities = question_vectors @ chunk_vectors.T
        rankings = [
            Ranking([pages[i] for i in np.argsort(-row)[:10]], float(row.max()))
            for row in similarities
        ]
        s = score(questions, rankings)
        columns = ["recall@1", "recall@5", "mrr@10", "threshold", "abstain_accuracy"]
        cells = [
            model,
            str(chunk_vectors.shape[1]),
            *(f"{s[c]:.2f}" for c in columns),
            f"{latency:.1f}",
        ]
        print("| " + " | ".join(cells) + " |")


def evaluate_system() -> None:
    """Upload the corpus to the running API and score its hybrid search."""
    page_by_id = _upload_corpus()
    questions = load_questions()
    rankings = []
    for q in questions:
        hits = _call("POST", "/search", {"query": q.text, "top_k": 10})
        rankings.append(
            # The answer step treats the most similar hit as the evidence, whatever its rank.
            Ranking(
                [page_by_id[h["document_id"]] for h in hits], max(h["similarity"] for h in hits)
            )
        )
    s = score(questions, rankings)
    print(json.dumps({k: round(v, 3) for k, v in s.items()}, indent=2))


def evaluate_answers() -> None:
    """Ask every question through the running API, language model included.

    Answerable: answered, citing at least one labelled page. Unanswerable: refused.
    """
    page_by_id = _upload_corpus()
    results: dict[str, list[bool]] = {"answerable": [], "unanswerable": []}
    for q in load_questions():
        reply = _call("POST", "/ask", {"question": q.text})
        cited = {page_by_id.get(c["document_id"]) for c in reply["citations"]}
        if q.pages:
            correct = reply["status"] == "answered" and bool(cited & set(q.pages))
        else:
            correct = reply["status"] == "insufficient_information"
        results["answerable" if q.pages else "unanswerable"].append(correct)
        if not correct:
            print(f"✗ {q.text} → {reply['status']}: {reply['answer'][:120]!r}")
    for kind, outcomes in results.items():
        print(f"{kind}: {sum(outcomes)}/{len(outcomes)} correct")


def _upload_corpus() -> dict[str, str]:
    """Upload the pages the API does not have yet; map each document id to its page."""
    documents = {d["filename"]: d for d in _call("GET", "/documents")}
    for path in sorted(CORPUS.glob("*.md")):
        if path.name not in documents:
            documents[path.name] = _upload(path)
    return {d["id"]: page_of(name) for name, d in documents.items()}


def _call(method: str, path: str, body: object = None) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        API + path, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())


def _upload(path: Path) -> Any:
    boundary = "evalboundary"
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: text/markdown\r\n\r\n".encode()
        + path.read_bytes()
        + f"\r\n--{boundary}--\r\n".encode()
    )
    request = urllib.request.Request(
        API + "/documents",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())


if __name__ == "__main__":
    modes = {"models": evaluate_models, "system": evaluate_system, "answers": evaluate_answers}
    modes[sys.argv[1]]()
