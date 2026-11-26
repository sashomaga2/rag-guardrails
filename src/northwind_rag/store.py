"""Knowledge base: Chroma (dense) + a lexical index (Elasticsearch / local BM25), fused with RRF,
then reranked by a cross-encoder (see rerank.py). Every search is limited to the chunks the
user's groups may read (see access.py).

Ingestion writes the same chunk ids to both stores and removes stale chunks from both.
Retrieval runs both searches in parallel; if the lexical side fails, it degrades to dense-only
instead of failing the chat. If the reranker fails or is slow, the RRF order is kept.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from northwind_rag.access import acl_metadata, can_read, chroma_filter, document_groups, split_front_matter
from northwind_rag.config import Settings
from northwind_rag.fusion import rrf_merge
from northwind_rag.lexical import ElasticsearchIndex, LexicalIndex, LocalBM25Index
from northwind_rag.rerank import Reranker

log = logging.getLogger(__name__)
_PLACEHOLDER_RE = re.compile(r"\[REDACTED:[a-z_]+\]")


@dataclass
class IngestReport:
    files: int = 0
    chunks: int = 0
    removed_stale: int = 0
    sources: list[str] = field(default_factory=list)


class KnowledgeBase:
    def __init__(self, settings: Settings, embeddings: Embeddings, persist_directory: Path | None = None,
                 lexical: LexicalIndex | None = None, reranker: Reranker | None = None):
        self.settings = settings
        self.reranker = reranker
        location = (
            {"host": settings.chroma_host, "port": settings.chroma_port}  # Chroma server (Kubernetes)
            if settings.chroma_host and persist_directory is None
            else {"persist_directory": str(persist_directory or settings.chroma_dir)}  # embedded (local)
        )
        self.vs = Chroma(
            # separate collections per embedding space: offline hash vectors != OpenAI vectors
            collection_name=f"{settings.collection}_{'offline' if settings.offline else 'online'}",
            embedding_function=embeddings,
            collection_configuration={"hnsw": {"space": "cosine"}},
            **location,
        )
        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "title"), ("##", "section")], strip_headers=False
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )
        self.lexical: LexicalIndex | None = lexical or self._build_lexical()

    def _build_lexical(self) -> LexicalIndex | None:
        s = self.settings
        if s.retrieval_mode != "hybrid":
            return None
        if s.elasticsearch_url:
            return ElasticsearchIndex(s.elasticsearch_url, s.elasticsearch_index, s.elasticsearch_api_key,
                                      s.lexical_min_should_match)
        # no Elasticsearch: in-process BM25, rebuilt from what is already in Chroma
        local = LocalBM25Index(min_should_match=s.lexical_min_should_match)
        stored = self.vs.get(include=["documents", "metadatas"])
        local.upsert([Document(page_content=t, metadata=m, id=i)
                      for i, t, m in zip(stored["ids"], stored["documents"], stored["metadatas"])])
        return local

    # -- ingestion ------------------------------------------------------------
    def chunk_file(self, path: Path) -> list[Document]:
        fields, text = split_front_matter(path.read_text(encoding="utf-8"))  # front matter is not indexed
        acl = acl_metadata(document_groups(fields))
        sections = self._header_splitter.split_text(text)
        chunks = self._splitter.split_documents(sections)
        out = []
        for i, c in enumerate(chunks):
            chunk_id = f"{path.stem}#{i}"
            c.metadata = {
                "source": path.name,
                "title": c.metadata.get("title", ""),
                "section": c.metadata.get("section", ""),
                "chunk_id": chunk_id,
                "content_hash": hashlib.sha256(c.page_content.encode()).hexdigest()[:16],
                **acl,
            }
            c.id = chunk_id
            out.append(c)
        return out

    def ingest(self, docs_dir: Path | None = None) -> IngestReport:
        """Upsert every markdown file into BOTH stores; delete stale chunks from both.

        Idempotent: unchanged chunks are overwritten in place, edited files lose chunks that no
        longer exist, deleted files disappear entirely. Safe to run from a Job and a CronJob.
        """
        docs_dir = docs_dir or self.settings.docs_dir
        report = IngestReport()
        paths = sorted(docs_dir.glob("*.md"))
        for path in paths:
            chunks = self.chunk_file(path)
            new_ids = [c.metadata["chunk_id"] for c in chunks]
            existing = self.vs.get(where={"source": path.name}, include=[])["ids"]
            stale = sorted(set(existing) - set(new_ids))
            if stale:
                self.vs.delete(ids=stale)
            if chunks:
                self.vs.add_documents(chunks, ids=new_ids)  # langchain-chroma upserts by id
            if self.lexical is not None:
                self.lexical.upsert(chunks)
                self.lexical.delete_source_except(path.name, new_ids)
            report.files += 1
            report.chunks += len(chunks)
            report.removed_stale += len(stale)
            report.sources.append(path.name)
        # files deleted from the docs folder: remove all their chunks
        present = {p.name for p in paths}
        gone = {m.get("source") for m in self.vs.get(include=["metadatas"])["metadatas"]} - present
        for source in sorted(filter(None, gone)):
            ids = self.vs.get(where={"source": source}, include=[])["ids"]
            self.vs.delete(ids=ids)
            if self.lexical is not None:
                self.lexical.delete_source_except(source, [])
            report.removed_stale += len(ids)
        log.info("ingested %s", report)
        return report

    def count(self) -> int:
        return len(self.vs.get(include=[])["ids"])

    def unlabelled_sample(self, limit: int = 50) -> int:
        """Chunks without an access label (ingested before access control) are hidden from every
        user (fail closed). Returns how many of the first `limit` chunks lack one."""
        metadatas = self.vs.get(limit=limit, include=["metadatas"])["metadatas"]
        return sum(1 for m in metadatas if not (m or {}).get("access_groups"))

    # -- retrieval ------------------------------------------------------------
    async def search(self, query: str, groups: frozenset[str] | set[str] = frozenset(), *,
                     wide: bool = False) -> list[Document]:
        """Only chunks readable by `groups` (+ public) are considered, in every store, before ranking.

        Stage 1 (recall): dense (above the relevance threshold) + lexical (minimum_should_match),
        fused with RRF. Stage 2 (precision): the cross-encoder rescores the top
        `rerank_candidates`, drops those below `rerank_min_score` and re-orders the rest.
        Finally at most N chunks per source file, top-k.

        `wide=True` (corrective retry) uses the larger retry_* limits. The relevance thresholds
        stay the same, so widening finds near-misses, never off-topic chunks."""
        s = self.settings
        fetch_k = s.retry_fetch_k if wide else s.fetch_k
        query = _PLACEHOLDER_RE.sub(" ", query)  # "[REDACTED:email]" carries no meaning for retrieval
        groups = frozenset(groups)
        dense_task = self.vs.asimilarity_search_with_relevance_scores(query, k=fetch_k, filter=chroma_filter(groups))
        if self.lexical is None:
            dense, lexical, mode = await dense_task, [], "dense"
        else:
            dense, lexical = await asyncio.gather(dense_task, self.lexical.search(query, fetch_k, groups),
                                                  return_exceptions=True)
            if isinstance(dense, BaseException):
                raise dense
            mode = "hybrid"
            if isinstance(lexical, BaseException):  # degrade, don't fail the chat
                log.warning("lexical search failed; using dense only", exc_info=lexical)
                lexical, mode = [], "dense_fallback"
        # defence in depth: the stores already filtered; anything else is a bug or a stale index
        dense, lexical = self._readable(dense, groups), self._readable(lexical, groups)

        dense = [(d, sc) for d, sc in sorted(dense, key=lambda x: x[1], reverse=True) if sc >= s.relevance_threshold]
        dense_score = {d.metadata["chunk_id"]: sc for d, sc in dense}
        lexical_score = {d.metadata["chunk_id"]: sc for d, sc in lexical}
        fused = rrf_merge(
            {"dense": [d for d, _ in dense], "lexical": [d for d, _ in lexical]},
            key=lambda d: d.metadata["chunk_id"],
            k=s.rrf_k,
            weights={"dense": s.rrf_dense_weight, "lexical": s.rrf_lexical_weight},
        )
        candidates: list[Document] = []
        for rank, f in enumerate(fused, start=1):
            doc: Document = f.item
            cid = doc.metadata["chunk_id"]
            doc.metadata.update(
                retrieval=mode,
                rrf_score=round(f.score, 5),
                rrf_rank=rank,
                dense_rank=f.ranks.get("dense"),
                lexical_rank=f.ranks.get("lexical"),
                score=round(float(dense_score[cid]), 4) if cid in dense_score else None,
                lexical_score=round(float(lexical_score[cid]), 3) if cid in lexical_score else None,
            )
            candidates.append(doc)

        candidates, reranked = await self._rerank(query, candidates,
                                                  s.retry_rerank_candidates if wide else s.rerank_candidates)

        top_k = s.retry_top_k if wide else s.top_k
        max_per_source = s.retry_max_chunks_per_source if wide else s.max_chunks_per_source
        results: list[Document] = []
        per_source: dict[str, int] = {}
        for doc in candidates:
            src = doc.metadata.get("source", "")
            if per_source.get(src, 0) >= max_per_source:
                continue
            per_source[src] = per_source.get(src, 0) + 1
            doc.metadata["reranked"] = reranked
            results.append(doc)
            if len(results) == top_k:
                break
        return results

    @staticmethod
    def _readable(hits: list, groups: frozenset[str]) -> list:
        kept = [(d, sc) for d, sc in hits if can_read(d.metadata, groups)]
        if len(kept) != len(hits):
            leaked = sorted({d.metadata.get("chunk_id") for d, _ in hits} - {d.metadata.get("chunk_id") for d, _ in kept})
            log.error("access filter: store returned %d unreadable chunk(s), dropped: %s", len(hits) - len(kept), leaked)
        return kept

    async def _rerank(self, query: str, candidates: list[Document], pool_size: int) -> tuple[list[Document], bool]:
        """Cross-encoder over the top `pool_size` RRF candidates. Returns (new order, reranked?).
        Chunks below the threshold are dropped, so an off-topic question can end with no context at all."""
        s = self.settings
        if self.reranker is None or not candidates:
            return candidates, False
        pool = candidates[:pool_size]
        try:
            async with asyncio.timeout(s.rerank_timeout_s):
                scores = await self.reranker.score(query, [d.page_content for d in pool])
        except Exception:  # incl. TimeoutError: degrade to the RRF order, don't fail the chat
            log.warning("reranker %s failed; keeping RRF order", self.reranker.name, exc_info=True)
            return candidates, False
        if len(scores) != len(pool):
            log.warning("reranker %s returned %d scores for %d chunks; keeping RRF order",
                        self.reranker.name, len(scores), len(pool))
            return candidates, False
        order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))  # ties keep the RRF order
        kept = []
        for position, i in enumerate(order, start=1):
            doc = pool[i]
            doc.metadata.update(rerank_score=round(scores[i], 4), rerank_rank=position, reranker=self.reranker.name)
            if scores[i] >= s.rerank_threshold:
                kept.append(doc)
        return kept, True

    async def ready(self) -> bool:
        return self.lexical is None or await self.lexical.ping()
