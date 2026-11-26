"""Hybrid retrieval: RRF math, BM25, Elasticsearch adapter (fake client), fusion in the KB."""

import shutil

import pytest
from langchain_core.documents import Document

from northwind_rag.config import Settings
from northwind_rag.fusion import rrf_merge
from northwind_rag.lexical import ES_MAPPINGS, ElasticsearchIndex, LocalBM25Index, required_matches
from northwind_rag.models import HashingEmbeddings
from northwind_rag.store import KnowledgeBase


def doc(cid: str, text: str, source: str | None = None, title: str = "") -> Document:
    return Document(page_content=text, metadata={"chunk_id": cid, "source": source or cid.split("#")[0] + ".md",
                                                 "title": title, "section": ""}, id=cid)


# ----------------------------------------------------------------------------------- RRF
def test_rrf_matches_worked_example():
    fused = rrf_merge({"dense": ["A", "B"], "lexical": ["C", "x", "A"]}, k=60)
    order = [f.item for f in fused]
    assert order[:3] == ["A", "C", "B"]
    a = next(f for f in fused if f.item == "A")
    assert a.score == pytest.approx(1 / 61 + 1 / 63) and a.ranks == {"dense": 1, "lexical": 3}


def test_rrf_weights_and_duplicates():
    fused = rrf_merge({"dense": ["A", "A", "B"], "lexical": ["B"]}, k=1, weights={"lexical": 3.0})
    b = next(f for f in fused if f.item == "B")
    assert b.ranks["dense"] == 2  # duplicate "A" did not push B down
    assert fused[0].item == "B"  # 1/3 + 3/2 beats A's 1/2


def test_rrf_is_deterministic_on_ties():
    fused = rrf_merge({"dense": ["A", "B"], "lexical": ["B", "A"]}, k=60)
    assert [f.item for f in fused] == ["A", "B"]  # equal scores; A appears first in the first list
    with pytest.raises(ValueError):
        rrf_merge({"x": ["A"]}, k=0)


# ----------------------------------------------------------------------------------- BM25
@pytest.mark.parametrize("n,need", [(1, 1), (2, 2), (3, 1), (4, 2), (5, 3), (10, 6)])
def test_minimum_should_match(n, need):
    assert required_matches(n, "2<60%") == need


async def test_bm25_ranks_exact_rare_terms_first():
    idx = LocalBM25Index()
    idx.upsert([
        doc("vpn#1", "If the error Gateway not reachable appears, check Wi-Fi and reconnect."),
        doc("vpn#2", "The VPN client disconnects every few minutes; update the client."),
        doc("net#0", "The office network gateway is managed by the network team."),
    ])
    hits = await idx.search("error Gateway not reachable", k=3)
    assert hits[0][0].metadata["chunk_id"] == "vpn#1"


async def test_bm25_single_common_word_is_not_enough():
    idx = LocalBM25Index()
    idx.upsert([doc("a#0", "Northwind uses GlobalProtect for VPN."), doc("b#0", "Northwind laptops are refreshed every 4 years.")])
    assert await idx.search("Does Northwind reimburse gym memberships?", k=5) == []  # 1 of 4 terms


async def test_bm25_upsert_replaces_and_delete_stale():
    idx = LocalBM25Index()
    idx.upsert([doc("p#0", "old text about printers"), doc("p#1", "toner cartridges")])
    idx.upsert([doc("p#0", "new text about scanners")])
    assert await idx.search("printers", k=5) == []
    assert idx.delete_source_except("p.md", keep_ids=["p#0"]) == 1 and idx.count() == 1


# --------------------------------------------------------------------- Elasticsearch adapter
class FakeIndices:
    def __init__(self):
        self.created = None

    def exists(self, index):
        return self.created is not None

    def create(self, index, settings, mappings):
        self.created = {"index": index, "settings": settings, "mappings": mappings}


class FakeES:
    def __init__(self, bulk_errors=False):
        self.indices = FakeIndices()
        self.calls = []
        self.bulk_errors = bulk_errors

    def bulk(self, operations, refresh):
        self.calls.append(("bulk", operations, refresh))
        items = [{"index": {"_id": op["index"]["_id"], **({"error": {"type": "mapper"}} if self.bulk_errors else {})}}
                 for op in operations[::2]]
        return {"errors": self.bulk_errors, "items": items}

    def delete_by_query(self, **kw):
        self.calls.append(("delete_by_query", kw))
        return {"deleted": 2}


class FakeAsyncES:
    def __init__(self):
        self.last = None

    async def search(self, **kw):
        self.last = kw
        return {"hits": {"hits": [{"_score": 7.5, "_source": {"content": "Use Hardware Refresh.", "title": "Laptop",
                                                              "section": "Request", "source": "laptop.md",
                                                              "chunk_id": "laptop#1"}}]}}

    async def ping(self):
        return True


def es_index(**kw):
    return ElasticsearchIndex("http://es:9200", "kb", client=FakeES(**kw), async_client=FakeAsyncES())


def test_es_creates_strict_mapping_and_bulk_upserts_by_chunk_id():
    idx = es_index()
    idx.upsert([doc("vpn#0", "GlobalProtect", title="VPN")])
    assert idx.es.indices.created["mappings"] == ES_MAPPINGS and ES_MAPPINGS["dynamic"] == "strict"
    _, ops, refresh = idx.es.calls[0]
    assert ops[0] == {"index": {"_index": "kb", "_id": "vpn#0"}} and ops[1]["title"] == "VPN"
    assert refresh == "wait_for"  # searchable as soon as the ingest job reports success


def test_es_bulk_errors_fail_the_ingest():
    with pytest.raises(RuntimeError, match="bulk upsert failed"):
        es_index(bulk_errors=True).upsert([doc("a#0", "x")])


def test_es_deletes_stale_chunks_of_a_source():
    idx = es_index()
    assert idx.delete_source_except("vpn.md", ["vpn#0", "vpn#1"]) == 2
    _, kw = idx.es.calls[-1]
    assert kw["query"] == {"bool": {"filter": [{"term": {"source": "vpn.md"}}],
                                    "must_not": [{"ids": {"values": ["vpn#0", "vpn#1"]}}]}}


async def test_es_search_query_and_parsing():
    idx = es_index()
    hits = await idx.search("Hardware Refresh catalog item", k=5)
    mm = idx.aes.last["query"]["multi_match"]
    assert mm["minimum_should_match"] == "2<60%" and "content.exact" in mm["fields"] and "title^2" in mm["fields"]
    d, score = hits[0]
    assert d.metadata["chunk_id"] == "laptop#1" and score == 7.5


# ------------------------------------------------------------------------- KB integration
class StaticLexical:
    def __init__(self, hits=None, fail=False):
        self.hits, self.fail = hits or [], fail

    def upsert(self, docs): ...
    def delete_source_except(self, source, keep_ids): return 0
    def count(self): return len(self.hits)
    async def ping(self): return not self.fail

    async def search(self, query, k):
        if self.fail:
            raise ConnectionError("es down")
        return self.hits


@pytest.fixture
def kb_factory(tmp_path):
    def make(lexical=None, **overrides) -> KnowledgeBase:
        settings = Settings(offline=True, chroma_dir=tmp_path / "chroma", **overrides)
        kb = KnowledgeBase(settings, HashingEmbeddings(), lexical=lexical)
        kb.ingest()
        return kb

    return make


async def test_lexical_only_hit_is_rescued_by_fusion(kb_factory):
    """Dense misses it (below threshold), BM25 finds the exact error string -> it reaches the LLM."""
    rescued = doc("vpn#9", "Error 0x80070057 means the GlobalProtect certificate is missing.", source="vpn.md")
    kb = kb_factory(lexical=StaticLexical([(rescued, 12.0)]), offline_min_relevance=0.99)  # dense finds nothing
    results = await kb.search("error 0x80070057")
    assert [d.metadata["chunk_id"] for d in results] == ["vpn#9"]
    md = results[0].metadata
    assert md["retrieval"] == "hybrid" and md["lexical_rank"] == 1 and md["dense_rank"] is None


async def test_both_lists_agreeing_ranks_first(kb_factory):
    kb = kb_factory()
    results = await kb.search("How often do engineers get a new laptop?")
    top = results[0].metadata
    assert top["chunk_id"].startswith("laptop-refresh") and top["dense_rank"] and top["lexical_rank"]


async def test_lexical_outage_degrades_to_dense(kb_factory):
    kb = kb_factory(lexical=StaticLexical(fail=True))
    results = await kb.search("How often do engineers get a new laptop?")
    assert results and all(d.metadata["retrieval"] == "dense_fallback" for d in results)
    assert await kb.ready() is False  # readiness reports it


async def test_dense_mode_has_no_lexical_index(kb_factory):
    kb = kb_factory(retrieval_mode="dense")
    assert kb.lexical is None
    assert (await kb.search("VPN client"))[0].metadata["retrieval"] == "dense"


async def test_deleted_file_is_removed_from_both_stores(tmp_path):
    docs = tmp_path / "docs"
    shutil.copytree(Settings().docs_dir, docs)
    settings = Settings(offline=True, chroma_dir=tmp_path / "chroma", docs_dir=docs)
    kb = KnowledgeBase(settings, HashingEmbeddings())
    kb.ingest()
    before = kb.lexical.count()
    (docs / "vpn.md").unlink()
    report = kb.ingest()
    assert report.removed_stale > 0
    assert not kb.vs.get(where={"source": "vpn.md"}, include=[])["ids"]
    assert kb.lexical.count() == before - report.removed_stale
    assert all(d.metadata["source"] != "vpn.md" for d in await kb.search("GlobalProtect VPN portal"))
