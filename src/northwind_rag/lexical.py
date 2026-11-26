"""Lexical (keyword / BM25) index: the second half of hybrid retrieval.

Dense vectors are good at meaning ("my VPN keeps dropping" ~ "disconnects every few minutes")
but weak at exact tokens: error messages, product codes, SKUs, catalog item names, rare words.
BM25 is the opposite. Running both and fusing with RRF (fusion.py) gets the best of each.

ElasticsearchIndex  production: english analyzer (stemming, stopwords) + an `exact` standard-
                    analyzer subfield for codes and error strings; title/section boosted.
LocalBM25Index      in-process BM25 with the same matching rule, for offline mode and tests.

Both apply a `minimum_should_match` rule. A chunk that only the lexical side found must match
a real share of the query terms, otherwise a single common word ("Northwind") would drag
unrelated chunks into the context and hide a genuine "I don't know".
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Protocol

from langchain_core.documents import Document

from northwind_rag.access import can_read, chunk_groups, readable_by
from northwind_rag.textutil import content_tokens

log = logging.getLogger(__name__)


class LexicalIndex(Protocol):
    def upsert(self, docs: list[Document]) -> None: ...
    def delete_source_except(self, source: str, keep_ids: list[str]) -> int: ...
    async def search(self, query: str, k: int, groups: frozenset[str] = frozenset()) -> list[tuple[Document, float]]:
        """Only chunks readable with `groups` (+ public) are searched (see access.py)."""
        ...
    def count(self) -> int: ...
    async def ping(self) -> bool: ...


def required_matches(n_terms: int, rule: str = "2<60%") -> int:
    """Elasticsearch `minimum_should_match` semantics for the subset we use: "N<P%" means
    all terms are required when there are N or fewer, otherwise P% (rounded down)."""
    m = re.fullmatch(r"(\d+)<(\d+)%", rule.strip())
    if not m:
        raise ValueError(f"unsupported minimum_should_match rule: {rule}")
    upto, pct = int(m.group(1)), int(m.group(2))
    if n_terms <= upto:
        return n_terms
    return max(1, math.floor(n_terms * pct / 100))


# ------------------------------------------------------------------------------- local BM25
class LocalBM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75, min_should_match: str = "2<60%") -> None:
        self.k1, self.b, self.msm = k1, b, min_should_match
        self._docs: dict[str, Document] = {}
        self._tf: dict[str, Counter] = {}
        self._len: dict[str, int] = {}
        self._df: Counter = Counter()

    @staticmethod
    def _fields(doc: Document) -> list[str]:
        md = doc.metadata
        # title/section counted twice: the same boost the Elasticsearch query applies
        return content_tokens(doc.page_content) + 2 * content_tokens(f"{md.get('title', '')} {md.get('section', '')}")

    def _remove(self, chunk_id: str) -> None:
        if chunk_id in self._tf:
            for term in self._tf[chunk_id]:
                self._df[term] -= 1
            del self._tf[chunk_id], self._len[chunk_id], self._docs[chunk_id]

    def upsert(self, docs: list[Document]) -> None:
        for d in docs:
            cid = d.metadata["chunk_id"]
            self._remove(cid)
            toks = self._fields(d)
            self._docs[cid], self._tf[cid], self._len[cid] = d, Counter(toks), len(toks)
            for term in self._tf[cid]:
                self._df[term] += 1

    def delete_source_except(self, source: str, keep_ids: list[str]) -> int:
        stale = [cid for cid, d in self._docs.items() if d.metadata.get("source") == source and cid not in set(keep_ids)]
        for cid in stale:
            self._remove(cid)
        return len(stale)

    async def search(self, query: str, k: int, groups: frozenset[str] = frozenset()) -> list[tuple[Document, float]]:
        terms = list(dict.fromkeys(content_tokens(query)))
        if not terms or not self._docs:
            return []
        need = required_matches(len(terms), self.msm)
        n, avg = len(self._docs), sum(self._len.values()) / len(self._docs)
        scored = []
        for cid, tf in self._tf.items():
            if not can_read(self._docs[cid].metadata, groups):  # filter before ranking
                continue
            matched = [t for t in terms if t in tf]
            if len(matched) < need:
                continue
            score = 0.0
            for t in matched:
                idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                f = tf[t]
                score += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self._len[cid] / avg))
            scored.append((self._docs[cid], score))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def count(self) -> int:
        return len(self._docs)

    async def ping(self) -> bool:
        return True


# ---------------------------------------------------------------------------- Elasticsearch
ES_SETTINGS = {"number_of_shards": 1, "number_of_replicas": 0}  # small corpus; raise replicas on a cluster
ES_MAPPINGS = {
    "dynamic": "strict",  # typos in field names fail loudly instead of creating new fields
    "properties": {
        "content": {"type": "text", "analyzer": "english", "fields": {"exact": {"type": "text", "analyzer": "standard"}}},
        "title": {"type": "text", "analyzer": "english"},
        "section": {"type": "text", "analyzer": "english"},
        "source": {"type": "keyword"},
        "chunk_id": {"type": "keyword"},
        "content_hash": {"type": "keyword"},
        "access_groups": {"type": "keyword"},  # who may read the chunk (array), see access.py
    },
}


def es_query(query: str, min_should_match: str, groups: frozenset[str] = frozenset()) -> dict:
    return {
        "bool": {
            "must": {
                "multi_match": {
                    "query": query,
                    "fields": ["content", "content.exact", "title^2", "section^2"],
                    "type": "most_fields",  # stemmed + exact both count: exact error strings rank higher
                    "minimum_should_match": min_should_match,
                }
            },
            # access control inside the query (a filter: no effect on scores, cached by ES)
            "filter": [{"terms": {"access_groups": readable_by(groups)}}],
        }
    }


class ElasticsearchIndex:
    """Writes with the sync client (ingest Job), searches with the async client (API)."""

    def __init__(self, url: str, index: str, api_key: str | None = None, min_should_match: str = "2<60%",
                 client=None, async_client=None) -> None:
        from elasticsearch import AsyncElasticsearch, Elasticsearch

        auth = {"api_key": api_key} if api_key else {}
        self.es = client or Elasticsearch(url, request_timeout=10, retry_on_timeout=True, max_retries=2, **auth)
        self.aes = async_client or AsyncElasticsearch(url, request_timeout=5, **auth)
        self.index = index
        self.msm = min_should_match
        self._mapping_checked = False

    def ensure_index(self) -> None:
        if not self.es.indices.exists(index=self.index):
            self.es.indices.create(index=self.index, settings=ES_SETTINGS, mappings=ES_MAPPINGS)
        elif not self._mapping_checked:
            # an index created before a new field (access_groups): adding fields is allowed
            self.es.indices.put_mapping(index=self.index, properties=ES_MAPPINGS["properties"])
        self._mapping_checked = True

    def upsert(self, docs: list[Document]) -> None:
        if not docs:
            return
        self.ensure_index()
        ops: list[dict] = []
        for d in docs:
            md = d.metadata
            ops.append({"index": {"_index": self.index, "_id": md["chunk_id"]}})
            ops.append({"content": d.page_content, "title": md.get("title", ""), "section": md.get("section", ""),
                        "source": md.get("source", ""), "chunk_id": md["chunk_id"], "content_hash": md.get("content_hash", ""),
                        "access_groups": sorted(chunk_groups(md))})
        resp = self.es.bulk(operations=ops, refresh="wait_for")
        if resp.get("errors"):
            failed = [i for i in resp["items"] if i.get("index", {}).get("error")]
            raise RuntimeError(f"Elasticsearch bulk upsert failed for {len(failed)} chunk(s): {failed[:2]}")

    def delete_source_except(self, source: str, keep_ids: list[str]) -> int:
        self.ensure_index()
        resp = self.es.delete_by_query(
            index=self.index,
            query={"bool": {"filter": [{"term": {"source": source}}], "must_not": [{"ids": {"values": keep_ids}}]}},
            refresh=True,
            conflicts="proceed",
        )
        return int(resp.get("deleted", 0))

    async def search(self, query: str, k: int, groups: frozenset[str] = frozenset()) -> list[tuple[Document, float]]:
        resp = await self.aes.search(index=self.index, query=es_query(query, self.msm, groups), size=k,
                                     source_excludes=["content_hash"])
        out = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            md = {"source": src["source"], "title": src.get("title", ""), "section": src.get("section", ""),
                  "chunk_id": src["chunk_id"], "access_groups": ",".join(src.get("access_groups") or [])}
            out.append((Document(page_content=src["content"], metadata=md, id=src["chunk_id"]), float(hit["_score"])))
        return out

    def count(self) -> int:
        if not self.es.indices.exists(index=self.index):
            return 0
        return int(self.es.count(index=self.index)["count"])

    async def ping(self) -> bool:
        try:
            return bool(await self.aes.ping())
        except Exception:
            return False
