"""Wires settings, models, knowledge base and graph together. Also the guarded streaming loop."""

from __future__ import annotations

import contextlib
import logging
from contextlib import aclosing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Iterable

from northwind_rag.config import Settings, get_settings
from northwind_rag.graph import RAGState, build_graph, summarize
from northwind_rag.guardrails.patterns import output_redactor
from northwind_rag.guardrails.streaming import ModerationMonitor, StreamRedactor, make_moderator
from northwind_rag.models import ModelSuite, build_suite
from northwind_rag.conversations import ConversationBus, LocalBus, build_bus
from northwind_rag.handoff import LiveAgentService, StubLiveAgentService
from northwind_rag.history import ConversationStore, InMemoryConversationStore, build_history
from northwind_rag.limits import CapacityExceeded, Limiter, LocalLimiter, build_limiter
from northwind_rag.store import KnowledgeBase

log = logging.getLogger(__name__)

MODERATION_REPLACEMENT = "I can't show that answer. Please contact the IT helpdesk at helpdesk@northwind.example."
BUSY_MESSAGE = "The assistant is busy right now. Please try again in a moment."


@dataclass
class Pipeline:
    settings: Settings
    suite: ModelSuite
    kb: KnowledgeBase
    graph: Any
    live_agent: LiveAgentService
    limiter: Limiter = field(default_factory=LocalLimiter)
    bus: ConversationBus = field(default_factory=LocalBus)
    history: ConversationStore = field(default_factory=InMemoryConversationStore)

    @classmethod
    def create(
        cls,
        settings: Settings | None = None,
        persist_directory: Path | None = None,
        live_agent: LiveAgentService | None = None,
        limiter: Limiter | None = None,
        bus: ConversationBus | None = None,
        history: ConversationStore | None = None,
    ) -> "Pipeline":
        settings = settings or get_settings()
        live_agent = live_agent or StubLiveAgentService()
        limiter = limiter or build_limiter(settings.redis_url, settings.lease_ttl_s)
        bus = bus or build_bus(settings.redis_url, settings.conversation_ttl_s)
        history = history or build_history(settings.database_url, settings.db_pool_min, settings.db_pool_max)
        suite = build_suite(settings, limiter)
        kb = KnowledgeBase(settings, suite.embeddings, persist_directory, reranker=suite.reranker)
        if settings.auto_ingest and kb.count() == 0:
            log.info("empty collection; ingesting %s", settings.docs_dir)
            kb.ingest()
        if (unlabelled := kb.unlabelled_sample()) > 0:
            log.warning("%d indexed chunk(s) have no access label and are hidden from everyone: "
                        "run `python -m northwind_rag.cli ingest` to re-index", unlabelled)
        graph = build_graph(suite, kb, settings, live_agent, bus)
        return cls(settings, suite, kb, graph, live_agent, limiter, bus, history)

    @contextlib.asynccontextmanager
    async def admit(self, user_id: str | None):
        """Admission control: global concurrent chats + concurrent streams per user.

        Raises CapacityExceeded; the API answers 503 + Retry-After.
        """
        s = self.settings
        async with contextlib.AsyncExitStack() as stack:
            if user_id:  # per-user first: one noisy user must not consume global slots while waiting
                await stack.enter_async_context(self.limiter.slot(f"user:{user_id}", s.max_streams_per_user, 0))
            await stack.enter_async_context(self.limiter.slot("chats", s.max_concurrent_chats, s.slot_wait_s))
            yield

    async def ready(self) -> dict[str, bool]:
        checks = {"limiter": await self.limiter.ping()}
        try:
            self.kb.count()
            checks["vector_store"] = True
        except Exception:
            checks["vector_store"] = False
        checks["lexical_index"] = await self.kb.ready()
        checks["database"] = await self.history.ping()  # reachable and migrated
        return checks

    @staticmethod
    def _inputs(question: str, conversation_id: str | None, user_id: str | None,
                groups: Iterable[str] | None) -> RAGState:
        return {"question": question, "conversation_id": conversation_id, "user_id": user_id,
                "user_groups": sorted(set(groups or ()))}

    async def ask(
        self, question: str, conversation_id: str | None = None, user_id: str | None = None,
        groups: Iterable[str] | None = None, **config: Any
    ) -> dict[str, Any]:
        """`groups`: the user's groups from the verified token. None/empty -> public documents only."""
        state: RAGState = await self.graph.ainvoke(self._inputs(question, conversation_id, user_id, groups),
                                                   config=config or None)
        return summarize(state)

    async def stream(
        self, question: str, conversation_id: str | None = None, user_id: str | None = None,
        groups: Iterable[str] | None = None, **config: Any
    ) -> AsyncIterator[dict[str, Any]]:
        """Run the graph and yield client events with streaming guardrails applied.

        Events:
          status    {"stage": ...}                  pipeline progress
          delta     {"text": ...}                   safe text to append
          retract   {"reason", "replacement"?}      remove everything shown so far
          blocked   {"answer", "label"}             input guard refused (nothing was streamed)
          final     {"status", "answer"}            answer without streaming (e.g. no context)
          verified  {"citations", "grounding_score"} streamed answer passed end-of-stream checks
          handoff   {"reason", "status", "ticket_id", "message", ...} forwarded to a live agent
          error     {"code": "busy"|"timeout", "message", "retry_after_s"} capacity/timeout
          done      {...summary}                    always last
        """
        s = self.settings
        redactor = StreamRedactor(output_redactor(s.allowed_email_domains), s.stream_holdback_chars)
        monitor = ModerationMonitor(make_moderator(self.suite), s.moderation_every_chars)
        final_state: RAGState = {"question": question}

        def moderation_retract(category: str) -> dict[str, Any]:
            final_state.update(status="blocked", final_answer=MODERATION_REPLACEMENT)
            return {"event": "retract", "data": {"reason": f"moderation:{category}", "replacement": MODERATION_REPLACEMENT}}

        stream = self.graph.astream(
            self._inputs(question, conversation_id, user_id, groups), stream_mode=["messages", "updates", "custom"], config=config or None
        )
        streamed_any = False
        try:
            async with aclosing(stream):  # leaving early closes the graph run -> no more tokens paid for
                async for mode, chunk in stream:
                    if mode == "custom":
                        yield {"event": "status", "data": chunk}
                        continue

                    if mode == "messages":
                        message, meta = chunk
                        if meta.get("langgraph_node") != "generate":  # never stream guard/judge tokens
                            continue
                        safe = redactor.feed(message.content if isinstance(message.content, str) else "")
                        if safe:
                            streamed_any = True
                            yield {"event": "delta", "data": {"text": safe}}
                        monitor.observe(redactor.released)
                        flagged = monitor.poll()
                        if flagged:
                            yield moderation_retract(flagged.category)
                            break
                        continue

                    # mode == "updates": one dict per finished node
                    stop = False
                    for node, update in chunk.items():
                        update = update or {}
                        final_state.update(update)
                        if node == "generate":
                            tail = redactor.flush()
                            if tail:
                                yield {"event": "delta", "data": {"text": tail}}
                            verdict = await monitor.final(redactor.released)
                            if not verdict.allowed:
                                yield moderation_retract(verdict.category)
                                stop = True
                        elif node == "refuse":
                            yield {"event": "blocked", "data": {"answer": update["final_answer"], "label": final_state.get("input_label")}}
                        elif node == "no_context":
                            yield {"event": "final", "data": {"status": "no_context", "answer": update["final_answer"]}}
                        elif node in ("check_output", "reretrieve"):  # reretrieve: only "ungrounded" (nothing new found)
                            status = update.get("status")
                            if status == "retrying":
                                yield {"event": "retract", "data": {"reason": "regenerating"}}
                                redactor.reset()
                                monitor.reset()
                            elif status == "answered":
                                yield {"event": "verified", "data": {"citations": update.get("citations", []), "grounding_score": update.get("grounding_score")}}
                            elif status == "ungrounded":
                                yield {"event": "retract", "data": {"reason": "ungrounded", "replacement": update["final_answer"]}}
                            elif status == "no_context":
                                yield {"event": "final", "data": {"status": "no_context", "answer": update["final_answer"]}}
                        elif node == "handoff":
                            yield {"event": "handoff", "data": {"reason": update["handoff_reason"], **update["handoff"]}}
                    if stop:
                        break
        except (CapacityExceeded, TimeoutError) as exc:
            # A model call found no free slot or timed out mid-run: fail closed, never half-checked.
            code = "busy" if isinstance(exc, CapacityExceeded) else "timeout"
            log.warning("stream aborted: %s", exc or code)
            if streamed_any:
                yield {"event": "retract", "data": {"reason": code, "replacement": BUSY_MESSAGE}}
            yield {"event": "error", "data": {"code": code, "message": BUSY_MESSAGE, "retry_after_s": 2}}
            final_state.update(status="error", final_answer=BUSY_MESSAGE)
        finally:
            monitor.cancel()
        yield {
            "event": "done",
            "data": {**summarize(final_state), "stream_redactions": redactor.findings, "moderation_checks": monitor.checks},
        }
