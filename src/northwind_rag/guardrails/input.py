"""Input guardrails: run before any retrieval output reaches the generator.

Layer 1 (deterministic, ~0 ms): length check, PII/secret redaction, injection regexes.
Layer 2 (LLM classifier, small model): catches paraphrased injections, harmful and off-topic asks.
Layer 2 is skipped when layer 1 already blocks, which saves cost and latency on obvious attacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from northwind_rag.config import Settings
from northwind_rag.guardrails.patterns import injection_hits, input_redactor, wants_live_agent
from northwind_rag.models import ModelSuite
from northwind_rag.prompts import INPUT_CLASSIFIER_SYSTEM
from northwind_rag.schemas import InputVerdict


@dataclass
class PreparedInput:
    sanitized: str
    pii_found: list[str] = field(default_factory=list)
    valid: bool = True


def prepare_input(question: str, settings: Settings) -> PreparedInput:
    """Deterministic step. Its output is the only version of the question used downstream."""
    q = (question or "").strip()
    if not q or len(q) > settings.max_question_chars:
        return PreparedInput(sanitized="", valid=False)
    sanitized, found = input_redactor(settings.allowed_email_domains).redact(q)
    return PreparedInput(sanitized=sanitized, pii_found=found)


@traceable(name="input_guard", run_type="chain")
async def classify_input(question: str, suite: ModelSuite) -> InputVerdict:
    hits = injection_hits(question)
    if hits:
        return InputVerdict(label="prompt_injection", reason=f"matched heuristic: {hits[0][:60]}")
    if wants_live_agent(question):
        return InputVerdict(label="agent_request", reason="user asked for a human agent")
    if suite.guard is None:  # offline: heuristics only
        return InputVerdict(label="safe", reason="offline mode: heuristics passed")
    classifier = suite.guard.with_structured_output(InputVerdict)
    messages = [
        SystemMessage(INPUT_CLASSIFIER_SYSTEM),
        HumanMessage(f"<user_message>\n{question}\n</user_message>"),
    ]
    return await suite.call("guard", lambda: classifier.ainvoke(messages))
