"""End-of-stream output guardrails: properties that need the complete answer.

- citation validation (deterministic): every cited id must be a retrieved chunk, and a factual
  answer must cite at least one source.
- groundedness (LLM, or lexical overlap offline): every claim must be supported by the context.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from northwind_rag.models import ModelSuite
from northwind_rag.prompts import GROUNDING_SYSTEM, IDK_ANSWER, format_context
from northwind_rag.schemas import ClaimCheck, GroundingVerdict
from northwind_rag.textutil import CITATION_RE, content_tokens, split_sentences, strip_citations


@dataclass
class CitationCheck:
    cited: list[str] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return bool(self.cited) and not self.invalid


def is_idk(answer: str) -> bool:
    return IDK_ANSWER.lower().rstrip(".") in answer.lower()


def validate_citations(answer: str, allowed_ids: list[str]) -> CitationCheck:
    cited = list(dict.fromkeys(CITATION_RE.findall(answer)))
    allowed = set(allowed_ids)
    return CitationCheck(cited=cited, invalid=[c for c in cited if c not in allowed])


def _lexical_grounding(answer: str, docs: list[Document], threshold: float = 0.75) -> GroundingVerdict:
    context_tokens = set(content_tokens(" ".join(d.page_content for d in docs)))
    claims = []
    for sentence in split_sentences(strip_citations(answer)):
        toks = content_tokens(sentence)
        if not toks:
            continue
        coverage = sum(t in context_tokens for t in toks) / len(toks)
        claims.append(ClaimCheck(claim=sentence, supported=coverage >= threshold))
    return GroundingVerdict(claims=claims, grounded=all(c.supported for c in claims))


@traceable(name="grounding_check", run_type="chain")
async def check_grounding(answer: str, docs: list[Document], suite: ModelSuite) -> GroundingVerdict:
    if suite.guard is None:
        return _lexical_grounding(answer, docs)
    checker = suite.guard.with_structured_output(GroundingVerdict)
    messages = [
        SystemMessage(GROUNDING_SYSTEM),
        HumanMessage(f"{format_context(docs)}\n\n<answer>\n{answer}\n</answer>"),
    ]
    return await suite.call("guard", lambda: checker.ainvoke(messages))
