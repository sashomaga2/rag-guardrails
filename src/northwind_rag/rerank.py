"""Second-stage reranking with a cross-encoder.

Retrieval is two stages:

    1. recall    dense (Chroma) + BM25, fused with RRF  -> `rerank_candidates` chunks (default 20)
    2. precision cross-encoder scores every (question, chunk) pair -> sort -> threshold -> top_k

Why a cross-encoder after RRF: the bi-encoder embeds the question and the chunk SEPARATELY, so
it can only compare two fixed vectors. A cross-encoder reads both texts together in one
transformer pass, so every question token attends to every chunk token ("reset" next to
"password", a negation, which product a version number belongs to). That is far more accurate,
and far too slow to run over the whole corpus, so it only re-orders the short candidate list.

Scores are made comparable across backends: every backend returns a relevance probability in
[0, 1] (sigmoid of the model's logit), so `rerank_min_score` means the same thing everywhere.

Backends (RAG_RERANKER):
    local   sentence-transformers CrossEncoder in the API process (pip install ".[rerank]")
    http    a Text Embeddings Inference server (POST /rerank); keeps torch out of the API image
    none    RRF order only (the previous behaviour)
Offline mode uses OverlapReranker: deterministic, keyless, same contract.

Failures degrade, they never fail the chat: if reranking errors or exceeds `rerank_timeout_s`,
the RRF order is used and the chunks are marked `reranked=False`.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
from typing import Protocol, Sequence

from northwind_rag.config import Settings
from northwind_rag.textutil import content_tokens

log = logging.getLogger(__name__)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


class Reranker(Protocol):
    name: str

    async def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Relevance in [0, 1] for each text, in input order."""
        ...

    async def warm_up(self) -> None:
        """Load the model / check the server before traffic arrives (no-op if nothing to do)."""
        ...


# ------------------------------------------------------------------------------- local model
class CrossEncoderReranker:
    """sentence-transformers CrossEncoder, loaded lazily on first use (not at import or start-up,
    so pods that never rerank never pay for it). Inference is CPU/GPU-bound, so it runs in a
    worker thread, at most `concurrency` batches at a time per process."""

    def __init__(self, model_name: str, *, max_length: int = 512, batch_size: int = 32,
                 concurrency: int = 2, device: str | None = None, model=None) -> None:
        self.name = f"local:{model_name}"
        self.model_name, self.max_length, self.batch_size, self.device = model_name, max_length, batch_size, device
        self._model = model  # injectable for tests
        self._load_task: asyncio.Task | None = None
        # A thread can't be cancelled: when a caller times out, its inference keeps running.
        # So the limit is taken INSIDE the worker thread, where it bounds the real CPU work.
        self._sem = threading.BoundedSemaphore(concurrency)

    async def _get_model(self):
        if self._model is None:
            if self._load_task is None:
                self._load_task = asyncio.ensure_future(asyncio.to_thread(self._load))
            task = self._load_task
            try:
                # shield: a request that times out while the model loads must not cancel the load
                self._model = await asyncio.shield(task)
            except BaseException:
                if task.done() and self._load_task is task:
                    self._load_task = None  # the load itself failed: a later request retries it
                raise
        return self._model

    def _load(self):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise RuntimeError('RAG_RERANKER=local needs sentence-transformers: pip install ".[rerank]"') from exc
        log.info("loading cross-encoder %s", self.model_name)
        return CrossEncoder(self.model_name, max_length=self.max_length, device=self.device)

    async def warm_up(self) -> None:
        try:
            await self._get_model()
        except Exception:  # missing extra / no network: start anyway, searches use the RRF order
            log.error("cross-encoder %s unavailable; reranking disabled until it loads", self.model_name,
                      exc_info=True)

    async def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        model = await self._get_model()
        pairs = [(query, t) for t in texts]

        def predict():
            with self._sem:
                return model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False,
                                     convert_to_numpy=True)

        scores = [float(s) for s in await asyncio.to_thread(predict)]
        # Most rerankers (ms-marco MiniLM, bge-reranker) emit logits; some model configs make
        # predict() return probabilities already. Normalise to [0, 1] either way.
        if any(s < 0.0 or s > 1.0 for s in scores):
            scores = [sigmoid(s) for s in scores]
        return scores


# ---------------------------------------------------------------------------- remote server
class HttpReranker:
    """Hugging Face Text Embeddings Inference (TEI) `/rerank` endpoint:
    POST {"query": q, "texts": [...], "raw_scores": true, "truncate": true}
    ->   [{"index": i, "score": logit}, ...]   (sorted by score; we restore input order)"""

    def __init__(self, url: str, *, model_name: str = "", timeout_s: float = 2.0, client=None) -> None:
        import httpx

        self.name = f"http:{model_name or url}"
        self.url = url.rstrip("/") + "/rerank"
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    async def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        r = await self._client.post(self.url, json={"query": query, "texts": list(texts),
                                                    "raw_scores": True, "truncate": True})
        r.raise_for_status()
        scores = [0.0] * len(texts)
        for item in r.json():
            scores[int(item["index"])] = sigmoid(float(item["score"]))
        return scores

    async def warm_up(self) -> None:
        try:
            await self.score("ping", ["pong"])
        except Exception:  # the server may start after us; requests degrade to RRF meanwhile
            log.warning("reranker %s not reachable yet", self.url, exc_info=True)

    async def close(self) -> None:
        await self._client.aclose()


# ------------------------------------------------------------------------- offline stand-in
class OverlapReranker:
    """Deterministic stand-in for offline mode and tests. Like a cross-encoder it looks at the
    question and the chunk TOGETHER: share of question terms the chunk contains, plus a bonus
    for question bigrams that appear in the same order. Not a model; just the same contract."""

    name = "offline:overlap"

    async def warm_up(self) -> None:
        pass

    async def score(self, query: str, texts: Sequence[str]) -> list[float]:
        q = content_tokens(query)
        q_terms, q_bigrams = set(q), set(zip(q, q[1:]))
        out = []
        for text in texts:
            t = content_tokens(text)
            terms, bigrams = set(t), set(zip(t, t[1:]))
            coverage = len(q_terms & terms) / len(q_terms) if q_terms else 0.0
            phrase = len(q_bigrams & bigrams) / len(q_bigrams) if q_bigrams else 0.0
            out.append(round(0.8 * coverage + 0.2 * phrase, 6))
        return out


def build_reranker(settings: Settings) -> Reranker | None:
    if settings.reranker == "none":
        return None
    if settings.offline:
        return OverlapReranker()
    if settings.reranker == "http":
        if not settings.reranker_url:
            raise ValueError("RAG_RERANKER=http needs RAG_RERANKER_URL")
        return HttpReranker(settings.reranker_url, model_name=settings.reranker_model,
                            timeout_s=settings.rerank_timeout_s)
    return CrossEncoderReranker(settings.reranker_model, max_length=settings.rerank_max_length,
                                concurrency=settings.rerank_concurrency)
