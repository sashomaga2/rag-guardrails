"""Retrieval-side guardrails against indirect prompt injection and secret leakage.

Retrieved text is untrusted: a wiki page can contain "ignore previous instructions...".
Chunks that look like injections are dropped (and reported); secrets in the rest are masked
before the generator ever sees them, so the model cannot repeat them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document

from northwind_rag.guardrails.patterns import context_redactor, injection_hits


@dataclass
class ScreenResult:
    kept: list[Document] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)


def screen_documents(docs: list[Document]) -> ScreenResult:
    redactor = context_redactor()
    result = ScreenResult()
    for d in docs:
        hits = injection_hits(d.page_content)
        if hits:
            result.dropped.append(
                {"chunk_id": d.metadata.get("chunk_id"), "reason": "indirect_prompt_injection", "pattern": hits[0][:60]}
            )
            continue
        text, found = redactor.redact(d.page_content)
        meta = dict(d.metadata)
        if found:
            meta["redacted"] = ",".join(sorted(set(found)))
        result.kept.append(Document(page_content=text, metadata=meta, id=d.id))
    return result
