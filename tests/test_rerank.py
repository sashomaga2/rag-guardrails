"""Second-stage reranking: backends (local cross-encoder, TEI over HTTP, offline stand-in) and the
reranking step inside KnowledgeBase.search (order, threshold, degradation, per-source cap)."""

import asyncio
import json
import threading
import time

import httpx
import pytest
from langchain_core.documents import Document

from northwind_rag.config import Settings
from northwind_rag.models import HashingEmbeddings
from northwind_rag.rerank import (CrossEncoderReranker, HttpReranker, OverlapReranker, build_reranker,
                                  sigmoid)
from northwind_rag.store import KnowledgeBase
from tests.test_hybrid import StaticLexical, doc


# ------------------------------------------------------------------------------- backends
class FakeCrossEncoder:
    """Same predict() signature as sentence_transformers.CrossEncoder."""

    def __init__(self, scores, delay_s=0.0):
        self.scores, self.delay_s, self.calls, self.active, self.max_active = scores, delay_s, [], 0, 0
        self._lock = threading.Lock()

    def predict(self, pairs, batch_size=32, show_progress_bar=False, convert_to_numpy=True):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.delay_s)
        with self._lock:
            self.active -= 1
        self.calls.append(pairs)
        return [self.scores[text] for _, text in pairs]


async def test_cross_encoder_logits_become_probabilities():
    model = FakeCrossEncoder({"relevant": 8.0, "unrelated": -9.0, "meh": 0.0})
    r = CrossEncoderReranker("fake", model=model)
    scores = await r.score("q", ["relevant", "unrelated", "meh"])
    assert scores == pytest.approx([sigmoid(8.0), sigmoid(-9.0), 0.5])
    assert model.calls == [[("q", "relevant"), ("q", "unrelated"), ("q", "meh")]]  # one batch, pairs


async def test_cross_encoder_probabilities_pass_through():
    r = CrossEncoderReranker("fake", model=FakeCrossEncoder({"a": 0.9, "b": 0.1}))
    assert await r.score("q", ["a", "b"]) == pytest.approx([0.9, 0.1])
    assert await r.score("q", []) == []


async def test_cross_encoder_loads_once_and_a_timeout_does_not_cancel_the_load(monkeypatch):
    loads = []

    def slow_load(self):
        loads.append(1)
        time.sleep(0.3)
        return FakeCrossEncoder({"a": 2.0})

    monkeypatch.setattr(CrossEncoderReranker, "_load", slow_load)
    r = CrossEncoderReranker("fake")
    with pytest.raises(TimeoutError):  # first request gives up while the model is still loading
        async with asyncio.timeout(0.05):
            await r.score("q", ["a"])
    results = await asyncio.gather(*(r.score("q", ["a"]) for _ in range(5)))
    assert loads == [1] and all(s == pytest.approx([sigmoid(2.0)]) for s in results)


async def test_failed_load_does_not_crash_start_up_and_is_retried(monkeypatch):
    attempts = []

    def flaky_load(self):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("no network")
        return FakeCrossEncoder({"a": 3.0})

    monkeypatch.setattr(CrossEncoderReranker, "_load", flaky_load)
    r = CrossEncoderReranker("fake")
    await r.warm_up()  # logs, doesn't raise
    assert await r.score("q", ["a"]) == pytest.approx([sigmoid(3.0)]) and len(attempts) == 2


async def test_cross_encoder_bounds_concurrent_inference():
    model = FakeCrossEncoder({"a": 1.0}, delay_s=0.05)
    r = CrossEncoderReranker("fake", model=model, concurrency=2)
    await asyncio.gather(*(r.score("q", ["a"]) for _ in range(6)))
    assert model.max_active == 2


async def test_http_reranker_restores_input_order():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["body"] = str(request.url), json.loads(request.content)
        # TEI returns results sorted by score, not in input order
        return httpx.Response(200, json=[{"index": 1, "score": 6.0}, {"index": 0, "score": -4.0}])

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    r = HttpReranker("http://reranker:80/", client=client)
    assert await r.score("reset password", ["vpn", "password reset"]) == pytest.approx([sigmoid(-4.0), sigmoid(6.0)])
    assert seen["url"] == "http://reranker/rerank"  # trailing slash handled (httpx drops the default :80)
    assert seen["body"] == {"query": "reset password", "texts": ["vpn", "password reset"],
                            "raw_scores": True, "truncate": True}


async def test_http_reranker_errors_propagate():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(503)))
    with pytest.raises(httpx.HTTPStatusError):
        await HttpReranker("http://reranker", client=client).score("q", ["a"])


async def test_overlap_stand_in_prefers_the_chunk_that_answers():
    scores = await OverlapReranker().score(
        "how long is my account locked",
        ["After 5 failed attempts your account is locked for 15 minutes.", "Order a new monitor in ServiceNow."])
    assert scores[0] > 0.5 > scores[1] == 0.0


def test_build_reranker_choice(tmp_path):
    assert build_reranker(Settings(offline=True, reranker="none")) is None
    assert isinstance(build_reranker(Settings(offline=True)), OverlapReranker)
    assert isinstance(build_reranker(Settings(offline=False, reranker="local")), CrossEncoderReranker)
    assert isinstance(build_reranker(Settings(offline=False, reranker="http", reranker_url="http://r")), HttpReranker)
    with pytest.raises(ValueError, match="RERANKER_URL"):
        build_reranker(Settings(offline=False, reranker="http"))
    with pytest.raises(ValueError, match="rerank_candidates"):
        Settings(rerank_candidates=2, top_k=4)


# ---------------------------------------------------------------------- inside the KB search
class ScriptedReranker:
    name = "scripted"

    def __init__(self, scores=None, fail=False, delay_s=0.0):
        self.scores, self.fail, self.delay_s, self.seen = scores or {}, fail, delay_s, []

    async def score(self, query, texts):
        self.seen.append(list(texts))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.fail:
            raise ConnectionError("reranker down")
        return [self.scores.get(t, 0.0) for t in texts]

    async def warm_up(self):
        pass


A = doc("a#0", "alpha chunk", source="a.md")
A2 = doc("a#1", "alpha second chunk", source="a.md")
A3 = doc("a#2", "alpha third chunk", source="a.md")
B = doc("b#0", "beta chunk", source="b.md")
C = doc("c#0", "gamma chunk", source="c.md")


@pytest.fixture
def make_kb(tmp_path):
    def make(hits, reranker, **overrides) -> KnowledgeBase:
        # dense finds nothing (threshold 0.99): the lexical hits ARE the RRF order, A first
        settings = Settings(offline=True, chroma_dir=tmp_path / "chroma", offline_min_relevance=0.99, **overrides)
        return KnowledgeBase(settings, HashingEmbeddings(), lexical=StaticLexical([(d, 10.0 - i) for i, d in enumerate(hits)]),
                             reranker=reranker)

    return make


def ids(docs: list[Document]) -> list[str]:
    return [d.metadata["chunk_id"] for d in docs]


async def test_cross_encoder_reorders_the_fused_candidates(make_kb):
    r = ScriptedReranker({"gamma chunk": 0.95, "beta chunk": 0.6, "alpha chunk": 0.3})
    results = await make_kb([A, B, C], r).search("question")
    assert ids(results) == ["c#0", "b#0", "a#0"]  # RRF said A, B, C
    top = results[0].metadata
    assert top["reranked"] and top["rerank_rank"] == 1 and top["rerank_score"] == 0.95 and top["rrf_rank"] == 3
    assert r.seen == [["alpha chunk", "beta chunk", "gamma chunk"]]  # one call with all candidates


async def test_chunks_below_the_threshold_are_dropped(make_kb):
    kb = make_kb([A, B, C], ScriptedReranker({"beta chunk": 0.9, "alpha chunk": 0.1, "gamma chunk": 0.0}),
                 offline_rerank_min_score=0.2)
    assert ids(await kb.search("question")) == ["b#0"]


async def test_nothing_relevant_means_no_context(make_kb):
    """An off-topic question whose candidates all score low returns no chunks -> no_context/handoff."""
    kb = make_kb([A, B], ScriptedReranker({}), offline_rerank_min_score=0.2)
    assert await kb.search("cafeteria menu") == []


async def test_only_the_top_candidates_are_reranked(make_kb):
    r = ScriptedReranker({"gamma chunk": 0.9, "beta chunk": 0.8, "alpha chunk": 0.7})
    results = await make_kb([A, B, C], r, rerank_candidates=2, top_k=2).search("question")
    assert r.seen == [["alpha chunk", "beta chunk"]] and ids(results) == ["b#0", "a#0"]


@pytest.mark.parametrize("reranker", [ScriptedReranker(fail=True), ScriptedReranker(delay_s=0.5)],
                         ids=["error", "timeout"])
async def test_reranker_failure_keeps_the_rrf_order(make_kb, reranker):
    results = await make_kb([A, B, C], reranker, rerank_timeout_s=0.05).search("question")
    assert ids(results) == ["a#0", "b#0", "c#0"]
    assert all(d.metadata["reranked"] is False for d in results)


async def test_per_source_cap_applies_after_reranking(make_kb):
    r = ScriptedReranker({"alpha chunk": 0.9, "alpha second chunk": 0.8, "alpha third chunk": 0.7, "beta chunk": 0.6})
    results = await make_kb([B, A, A2, A3], r, max_chunks_per_source=2).search("question")
    assert ids(results) == ["a#0", "a#1", "b#0"]  # third a.md chunk skipped for diversity


async def test_wide_search_relaxes_the_limits_for_the_corrective_retry(make_kb):
    r = ScriptedReranker({"alpha chunk": 0.9, "alpha second chunk": 0.85, "alpha third chunk": 0.8,
                          "beta chunk": 0.7, "gamma chunk": 0.01})
    kb = make_kb([A, A2, A3, B, C], r, top_k=2, max_chunks_per_source=1, offline_rerank_min_score=0.2,
                 retry_top_k=5, retry_max_chunks_per_source=3)
    assert ids(await kb.search("question")) == ["a#0", "b#0"]
    # neighbouring chunks of the same file come back; the relevance threshold still drops gamma
    assert ids(await kb.search("question", wide=True)) == ["a#0", "a#1", "a#2", "b#0"]


def test_retry_limits_can_only_widen():
    with pytest.raises(ValueError, match="retry_top_k"):
        Settings(offline=True, top_k=4, retry_top_k=2)
    with pytest.raises(ValueError, match="retry_rerank_candidates"):
        Settings(offline=True, retry_top_k=8, retry_rerank_candidates=4)


async def test_pipeline_answers_from_the_reranked_chunks(pipeline):
    question = "How long is my account locked after failed logins?"
    docs = await pipeline.kb.search(question)
    assert docs and all(d.metadata["reranked"] and d.metadata["reranker"] == "offline:overlap" for d in docs)
    assert [d.metadata["rerank_rank"] for d in docs] == sorted(d.metadata["rerank_rank"] for d in docs)
    result = await pipeline.ask(question)
    assert result["status"] == "answered" and "15 minutes" in result["answer"]
