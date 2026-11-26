"""Model factory.

Online: real chat models via `init_chat_model` (any provider string, e.g. "openai:gpt-4.1-mini",
"anthropic:claude-sonnet-4-5", "azure_openai:..."), OpenAI embeddings.

Offline: deterministic stand-ins with the same interfaces, so the graph, the streaming
guard, the API and the evaluation harness run in CI without network or keys.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterator, Literal, TypeVar

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from northwind_rag.config import Settings
from northwind_rag.limits import Limiter, LocalLimiter

T = TypeVar("T")
from northwind_rag.prompts import IDK_ANSWER
from northwind_rag.textutil import content_tokens, split_sentences


Role = Literal["generator", "guard"]


@dataclass
class CallPolicy:
    """Every LLM call goes through here: a concurrency slot per role + a hard timeout.

    Errors propagate on purpose (fail closed): a guard that times out must never let an
    unchecked request through. The API turns CapacityExceeded/TimeoutError into 503s.
    """

    limiter: Limiter = field(default_factory=LocalLimiter)
    limits: dict[str, int] = field(default_factory=lambda: {"generator": 20, "guard": 40})
    timeouts: dict[str, float] = field(default_factory=lambda: {"generator": 60.0, "guard": 20.0})
    wait_s: float = 5.0

    @classmethod
    def from_settings(cls, settings: Settings, limiter: Limiter) -> "CallPolicy":
        return cls(
            limiter=limiter,
            limits={"generator": settings.generator_concurrency, "guard": settings.guard_concurrency},
            timeouts={"generator": settings.generation_timeout_s, "guard": settings.llm_timeout_s},
            wait_s=settings.slot_wait_s,
        )

    async def run(self, role: Role, call: Callable[[], Awaitable[T]]) -> T:
        async with self.limiter.slot(f"llm:{role}", self.limits[role], self.wait_s):
            async with asyncio.timeout(self.timeouts[role]):
                return await call()


@dataclass
class ModelSuite:
    chat: BaseChatModel  # streaming generator
    guard: BaseChatModel | None  # fast classifier model (None offline -> heuristics)
    judge: BaseChatModel | None  # evaluation judge (None offline)
    embeddings: Embeddings
    offline: bool
    policy: CallPolicy = field(default_factory=CallPolicy)

    async def call(self, role: Role, call: Callable[[], Awaitable[T]]) -> T:
        return await self.policy.run(role, call)


def build_suite(settings: Settings, limiter: Limiter | None = None) -> ModelSuite:
    policy = CallPolicy.from_settings(settings, limiter or LocalLimiter())
    if settings.offline:
        return ModelSuite(
            chat=ExtractiveChatModel(),
            guard=None,
            judge=None,
            embeddings=HashingEmbeddings(),
            offline=True,
            policy=policy,
        )

    from langchain.chat_models import init_chat_model
    from langchain_openai import OpenAIEmbeddings

    return ModelSuite(
        chat=init_chat_model(settings.chat_model, temperature=0).with_config(tags=["generator"]),
        guard=init_chat_model(settings.guard_model, temperature=0).with_config(tags=["guard"]),
        judge=init_chat_model(settings.judge_model, temperature=0).with_config(tags=["judge"]),
        embeddings=OpenAIEmbeddings(model=settings.embedding_model),
        offline=False,
        policy=policy,
    )


# ---------------------------------------------------------------------------
# Offline embeddings: hashed unigrams + bigrams, sublinear tf, L2-normalised.
# Lexical, but good enough for meaningful retrieval over a small corpus.
# ---------------------------------------------------------------------------
class HashingEmbeddings(Embeddings):
    def __init__(self, dim: int = 2048) -> None:
        self.dim = dim

    def _bucket(self, feature: str) -> int:
        return int.from_bytes(hashlib.md5(feature.encode()).digest()[:4], "little") % self.dim

    def _embed(self, text: str) -> list[float]:
        toks = content_tokens(text)
        feats = Counter(toks) + Counter(f"{a}_{b}" for a, b in zip(toks, toks[1:]))
        vec = [0.0] * self.dim
        for f, c in feats.items():
            vec[self._bucket(f)] += 1.0 + math.log(c)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


# ---------------------------------------------------------------------------
# Offline chat model: extractive QA over the <context> block of the prompt.
# Implements _stream and reports tokens to callbacks, so LangGraph's
# stream_mode="messages" sees real token-by-token streaming.
# ---------------------------------------------------------------------------
_DOC_RE = re.compile(r'<doc id="([^"]+)"[^>]*>\n?(.*?)\n?</doc>', re.DOTALL)


class ExtractiveChatModel(BaseChatModel):
    token_delay_s: float = 0.0
    max_sentences: int = 2

    @property
    def _llm_type(self) -> str:
        return "offline-extractive"

    def _answer(self, messages: list[BaseMessage]) -> str:
        prompt = messages[-1].content if isinstance(messages[-1].content, str) else str(messages[-1].content)
        question = prompt.rsplit("Question:", 1)[-1].strip()
        q = set(content_tokens(question))
        scored: list[tuple[float, int, str, str]] = []
        for order, (doc_id, body) in enumerate(_DOC_RE.findall(prompt)):
            for sent in split_sentences(body):
                toks = content_tokens(sent)
                overlap = len(q & set(toks))
                # 1 shared term is enough in the best-ranked chunk (retrieval already vouched for it)
                if overlap >= 2 or (overlap == 1 and (len(q) <= 2 or order == 0)):
                    scored.append((overlap / math.sqrt(len(toks) or 1), -order, sent, doc_id))
        if not scored:
            return IDK_ANSWER
        scored.sort(reverse=True)
        best = scored[: self.max_sentences]
        return " ".join(f"{s.rstrip('.')} [{doc_id}]." for _, _, s, doc_id in best)

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(self._answer(messages)))])

    def _stream(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> Iterator[ChatGenerationChunk]:
        for token in re.findall(r"\S+\s*", self._answer(messages)):
            if self.token_delay_s:
                time.sleep(self.token_delay_s)
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk
