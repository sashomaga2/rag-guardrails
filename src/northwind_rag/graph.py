r"""The RAG pipeline as a LangGraph state machine.

    START -> prepare --+--> classify_input --+--> gate --> refuse -------------------------> END
                       |                     |       |--> handoff (user asked) ----------> END
                       +--> retrieve --------+       |--> no_context --> handoff --------> END
                                                     \--> generate --> check_output -----> END
                                                              ^              |  \--> handoff --> END
                                                              +-- retry -----+    (no answer / unverified)

- prepare:        deterministic validation + PII redaction (only the sanitized question flows on)
- classify_input  and retrieve run in PARALLEL: the LLM guard adds no latency to retrieval.
- gate:           join point; routes on the guard verdict and on whether any context survived.
- retrieve:       Chroma search + indirect-injection screening + secret masking of chunks.
- generate:       the only node whose tokens are streamed to the user.
- check_output:   redaction, citation validation, groundedness; one corrective retry.
- handoff:        forwards the (sanitized) conversation to a human via LiveAgentService when the
                  user asks for a person, the KB has no answer, or no verified answer was produced.
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.documents import Document
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from northwind_rag.config import Settings
from northwind_rag.conversations import ConversationBus, LocalBus
from northwind_rag.guardrails.documents import screen_documents
from northwind_rag.guardrails.input import classify_input, prepare_input
from northwind_rag.guardrails.output import check_grounding, is_idk, validate_citations
from northwind_rag.guardrails.patterns import output_redactor
from northwind_rag.handoff import (
    HANDOFF_UNAVAILABLE,
    HandoffReason,
    HandoffRequest,
    LiveAgentService,
    StubLiveAgentService,
)
from northwind_rag.models import ModelSuite
from northwind_rag.prompts import (
    BLOCKED_ANSWERS,
    IDK_ANSWER,
    UNVERIFIED_ANSWER,
    UNVERIFIED_SHORT,
    generator_messages,
)
from northwind_rag.store import KnowledgeBase

log = logging.getLogger(__name__)

Status = Literal["answered", "blocked", "no_context", "ungrounded", "retrying", "handed_off", "error"]


class RAGState(TypedDict, total=False):
    # input
    question: str
    conversation_id: str | None
    user_id: str | None
    # guardrail results
    sanitized_question: str
    input_valid: bool
    input_pii: list[str]
    input_label: str
    input_reason: str
    # retrieval
    documents: list[Document]
    dropped_documents: list[dict]
    # generation
    answer: str
    attempts: int
    feedback: str | None
    # verification
    citations: list[str]
    grounding_score: float
    unsupported_claims: list[str]
    output_redactions: Annotated[list[str], operator.add]
    # handoff
    handoff_reason: HandoffReason | None
    handoff: dict | None
    # result
    status: Status
    final_answer: str


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)


def build_graph(
    suite: ModelSuite,
    kb: KnowledgeBase,
    settings: Settings,
    live_agent: LiveAgentService | None = None,
    bus: ConversationBus | None = None,
):
    redact_output = output_redactor(settings.allowed_email_domains)
    live_agent = live_agent or StubLiveAgentService()
    bus = bus or LocalBus()

    def stage(name: str, **extra: Any) -> None:
        get_stream_writer()({"stage": name, **extra})  # no-op unless streaming with "custom"

    # -- nodes -------------------------------------------------------------------
    def prepare(state: RAGState) -> RAGState:
        stage("checking_input")
        prepared = prepare_input(state["question"], settings)
        return {
            "sanitized_question": prepared.sanitized,
            "input_valid": prepared.valid,
            "input_pii": prepared.pii_found,
            "attempts": 0,
            "output_redactions": [],
        }

    async def classify(state: RAGState) -> RAGState:
        if not state["input_valid"]:
            return {"input_label": "invalid", "input_reason": "empty or too long"}
        verdict = await classify_input(state["sanitized_question"], suite)
        return {"input_label": verdict.label, "input_reason": verdict.reason}

    async def retrieve(state: RAGState) -> RAGState:
        if not state["input_valid"]:
            return {"documents": [], "dropped_documents": []}
        stage("retrieving")
        docs = await kb.search(state["sanitized_question"])
        screened = screen_documents(docs)
        return {"documents": screened.kept, "dropped_documents": screened.dropped}

    def gate(state: RAGState) -> RAGState:
        return {}

    def route_after_gate(state: RAGState) -> str:
        if state.get("input_label") == "agent_request":
            return "handoff"
        if state.get("input_label") != "safe":
            return "refuse"
        if not state.get("documents"):
            return "no_context"
        return "generate"

    def refuse(state: RAGState) -> RAGState:
        label = state.get("input_label", "invalid")
        return {"status": "blocked", "final_answer": BLOCKED_ANSWERS.get(label, BLOCKED_ANSWERS["harmful"])}

    def no_context(state: RAGState) -> RAGState:
        reason = "no_answer" if settings.handoff_on_no_answer else None
        return {"status": "no_context", "final_answer": IDK_ANSWER, "handoff_reason": reason}

    async def generate(state: RAGState) -> RAGState:
        attempt = state.get("attempts", 0) + 1
        stage("generating", attempt=attempt)
        messages = generator_messages(state["sanitized_question"], state["documents"], state.get("feedback"))
        response = await suite.call("generator", lambda: suite.chat.ainvoke(messages))
        return {"answer": _text(response.content).strip(), "attempts": attempt}

    async def check_output(state: RAGState) -> RAGState:
        stage("verifying")
        answer, redactions = redact_output.redact(state["answer"])
        if is_idk(answer):
            reason = "no_answer" if settings.handoff_on_no_answer else None
            return {"status": "no_context", "final_answer": IDK_ANSWER, "output_redactions": redactions, "handoff_reason": reason}

        docs = state["documents"]
        citations = validate_citations(answer, [d.metadata["chunk_id"] for d in docs])
        grounding = await check_grounding(answer, docs, suite)
        result: RAGState = {
            "answer": answer,
            "citations": citations.cited,
            "grounding_score": round(grounding.score, 3),
            "unsupported_claims": grounding.unsupported,
            "output_redactions": redactions,
        }
        if citations.valid and grounding.grounded:
            return {**result, "status": "answered", "final_answer": answer, "feedback": None}

        problems = []
        if not citations.cited:
            problems.append("it cited no sources")
        if citations.invalid:
            problems.append(f"it cited ids that are not in the context: {citations.invalid}")
        if grounding.unsupported:
            problems.append(f"these claims are not supported: {grounding.unsupported}")
        if state["attempts"] < settings.max_generation_attempts:
            return {**result, "status": "retrying", "feedback": "; ".join(problems)}
        if settings.handoff_on_unverified:
            return {**result, "status": "ungrounded", "final_answer": UNVERIFIED_SHORT, "handoff_reason": "unverified_answer"}
        return {**result, "status": "ungrounded", "final_answer": UNVERIFIED_ANSWER}

    def route_after_check(state: RAGState) -> str:
        if state["status"] == "retrying":
            return "generate"
        return "handoff" if state.get("handoff_reason") else END

    def route_to_handoff(state: RAGState) -> str:
        return "handoff" if state.get("handoff_reason") else END

    async def handoff(state: RAGState) -> RAGState:
        stage("handing_off")
        reason: HandoffReason = state.get("handoff_reason") or "user_requested"
        request = HandoffRequest(
            reason=reason,
            question=state.get("sanitized_question", ""),
            conversation_id=state.get("conversation_id"),
            user_id=state.get("user_id"),
            retrieved_sources=[d.metadata["chunk_id"] for d in state.get("documents", [])],
            draft_answer=state.get("answer"),
            unsupported_claims=state.get("unsupported_claims", []),
            input_pii_redacted=state.get("input_pii", []),
        )
        try:
            result = await live_agent.handoff(request)
        except Exception:  # a broken integration must never break the chat
            log.exception("live agent handoff failed")
            result = HANDOFF_UNAVAILABLE
        conversation_id = state.get("conversation_id")
        if conversation_id and result.status != "unavailable":
            # From now on a human handles this conversation: user messages go to the desk, agent
            # replies arrive through the bus on whichever pod holds the user's WebSocket.
            # (Ownership was already claimed by the API before the graph ran.)
            await bus.set_mode(conversation_id, "agent")
            await bus.publish(conversation_id, "status", {"state": "waiting_for_agent", **result.to_dict()})
        if reason == "user_requested":
            return {"status": "handed_off", "handoff_reason": reason, "handoff": result.to_dict(), "final_answer": result.message}
        prefix = state.get("final_answer", "")
        return {"handoff_reason": reason, "handoff": result.to_dict(), "final_answer": f"{prefix} {result.message}".strip()}

    # -- wiring ------------------------------------------------------------------
    g = StateGraph(RAGState)
    g.add_node("prepare", prepare)
    g.add_node("classify_input", classify)
    g.add_node("retrieve", retrieve)
    g.add_node("gate", gate)
    g.add_node("refuse", refuse)
    g.add_node("no_context", no_context)
    g.add_node("generate", generate)
    g.add_node("check_output", check_output)
    g.add_node("handoff", handoff)

    g.add_edge(START, "prepare")
    g.add_edge("prepare", "classify_input")
    g.add_edge("prepare", "retrieve")
    g.add_edge(["classify_input", "retrieve"], "gate")  # waits for both branches
    g.add_conditional_edges("gate", route_after_gate, ["refuse", "handoff", "no_context", "generate"])
    g.add_edge("refuse", END)
    g.add_conditional_edges("no_context", route_to_handoff, ["handoff", END])
    g.add_edge("generate", "check_output")
    g.add_conditional_edges("check_output", route_after_check, ["generate", "handoff", END])
    g.add_edge("handoff", END)
    return g.compile()


def summarize(state: RAGState) -> dict[str, Any]:
    """Public, serialisable view of a finished run (used by the API and the evals)."""
    docs = state.get("documents", [])
    return {
        "answer": state.get("final_answer", ""),
        "status": state.get("status"),
        "question": state.get("sanitized_question", ""),  # redacted: the only version that is stored
        "citations": state.get("citations", []),
        "sources": [d.metadata["chunk_id"] for d in docs],
        "contexts": [d.page_content for d in docs],
        "input_label": state.get("input_label"),
        "input_pii_redacted": state.get("input_pii", []),
        "dropped_documents": state.get("dropped_documents", []),
        "output_redactions": state.get("output_redactions", []),
        "grounding_score": state.get("grounding_score"),
        "unsupported_claims": state.get("unsupported_claims", []),
        "attempts": state.get("attempts", 0),
        "handoff": state.get("handoff"),
    }
