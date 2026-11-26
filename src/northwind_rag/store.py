"""Knowledge base: Chroma (dense) + a lexical index (Elasticsearch / local BM25), fused with RRF.

Ingestion writes the same chunk ids to both stores and removes stale chunks from both.
Retrieval runs both searches in parallel; if the lexical side fails, it degrades to dense-only
instead of failing the chat.
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

from northwind_rag.config import Settings
from northwind_rag.fusion import rrf_merge
from northwind_rag.lexical import ElasticsearchIndex, LexicalIndex, LocalBM25Index

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
                 lexical: LexicalIndex | None = None):
        self.settings = settings
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
        text = path.read_text(encoding="utf-8")
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

    # -- retrieval ------------------------------------------------------------
    async def search(self, query: str) -> list[Document]:
        """Hybrid: dense (above the relevance threshold) + lexical (minimum_should_match),
        fused with RRF, at most N chunks per source file, top-k."""
        s = self.settings
        query = _PLACEHOLDER_RE.sub(" ", query)  # "[REDACTED:email]" carries no meaning for retrieval
        dense_task = self.vs.asimilarity_search_with_relevance_scores(query, k=s.fetch_k)
        if self.lexical is None:
            dense, lexical, mode = await dense_task, [], "dense"
        else:
            dense, lexical = await asyncio.gather(dense_task, self.lexical.search(query, s.fetch_k),
                                                  return_exceptions=True)
            if isinstance(dense, BaseException):
                raise dense
            mode = "hybrid"
            if isinstance(lexical, BaseException):  # degrade, don't fail the chat
                log.warning("lexical search failed; using dense only", exc_info=lexical)
                lexical, mode = [], "dense_fallback"

        dense = [(d, sc) for d, sc in sorted(dense, key=lambda x: x[1], reverse=True) if sc >= s.relevance_threshold]
        dense_score = {d.metadata["chunk_id"]: sc for d, sc in dense}
        lexical_score = {d.metadata["chunk_id"]: sc for d, sc in lexical}
        fused = rrf_merge(
            {"dense": [d for d, _ in dense], "lexical": [d for d, _ in lexical]},
            key=lambda d: d.metadata["chunk_id"],
            k=s.rrf_k,
            weights={"dense": s.rrf_dense_weight, "lexical": s.rrf_lexical_weight},
        )

        results: list[Document] = []
        per_source: dict[str, int] = {}
        for f in fused:
            doc: Document = f.item
            src = doc.metadata.get("source", "")
            if per_source.get(src, 0) >= s.max_chunks_per_source:
                continue
            per_source[src] = per_source.get(src, 0) + 1
            cid = doc.metadata["chunk_id"]
            doc.metadata.update(
                retrieval=mode,
                rrf_score=round(f.score, 5),
                dense_rank=f.ranks.get("dense"),
                lexical_rank=f.ranks.get("lexical"),
                score=round(float(dense_score[cid]), 4) if cid in dense_score else None,
                lexical_score=round(float(lexical_score[cid]), 3) if cid in lexical_score else None,
            )
            results.append(doc)
            if len(results) == s.top_k:
                break
        return results

    async def ready(self) -> bool:
        return self.lexical is None or await self.lexical.ping()
