"""Central configuration. Every value can be overridden with an env var prefixed RAG_."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
    )

    # --- Models -------------------------------------------------------------
    # offline=True swaps every model for deterministic local stand-ins so the
    # whole pipeline (and the test suite) runs without API keys.
    offline: bool = False
    chat_model: str = "openai:gpt-4.1-mini"  # generator (streams to the user)
    guard_model: str = "openai:gpt-4.1-nano"  # input / output classifiers: small + fast
    judge_model: str = "openai:gpt-4.1"  # evaluation judge: separate, stronger model
    embedding_model: str = "text-embedding-3-small"

    # --- Knowledge base -----------------------------------------------------
    docs_dir: Path = PROJECT_ROOT / "data" / "docs"
    static_dir: Path = PROJECT_ROOT / "static"
    auto_ingest: bool = True  # ingest on startup if empty; False in Kubernetes (a Job does it)
    chroma_dir: Path = PROJECT_ROOT / ".chroma"
    collection: str = "northwind_kb"
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- Retrieval ----------------------------------------------------------
    top_k: int = 4
    fetch_k: int = 12
    min_relevance: float = 0.25  # cosine relevance; below this a chunk is ignored (tune with evals)
    offline_min_relevance: float = 0.12  # hashed lexical vectors score lower than neural embeddings
    max_chunks_per_source: int = 2  # diversity: avoid 4 chunks from one file

    # --- Hybrid retrieval (dense + BM25, fused with Reciprocal Rank Fusion) -----
    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"
    elasticsearch_url: str | None = None  # unset -> in-process BM25 (offline / local)
    elasticsearch_api_key: str | None = None
    elasticsearch_index: str = "northwind_kb"
    lexical_min_should_match: str = "2<60%"  # <=2 terms: all must match, else 60% of them
    rrf_k: int = 60  # RRF damping constant (60 is the value from the original paper)
    rrf_dense_weight: float = 1.0
    rrf_lexical_weight: float = 1.0

    # --- Reranking (cross-encoder over the fused candidates; see rerank.py) ---------
    reranker: Literal["local", "http", "none"] = "local"  # offline mode always uses the overlap stand-in
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # fast on CPU; BAAI/bge-reranker-v2-m3 for multilingual
    reranker_url: str | None = None  # http: Text Embeddings Inference server, e.g. http://reranker:80
    rerank_candidates: int = 20  # fused chunks sent to the cross-encoder (recall stage size)
    rerank_min_score: float = 0.05  # relevance probability; below it a chunk is dropped (tune with evals)
    offline_rerank_min_score: float = 0.2  # the overlap stand-in scores on a different scale
    rerank_timeout_s: float = 2.0  # slower than this -> keep the RRF order instead
    rerank_max_length: int = 512  # tokens per (question, chunk) pair
    rerank_concurrency: int = 2  # concurrent local inference batches per process (CPU-bound)

    # --- Guardrails ---------------------------------------------------------
    max_question_chars: int = 2000
    max_generation_attempts: int = 2  # 1 retry when groundedness check fails
    stream_holdback_chars: int = 128  # must be >= patterns.MAX_MATCH_CHARS
    moderation_every_chars: int = 240  # run streaming moderation every N chars
    allowed_email_domains: list[str] = Field(default_factory=lambda: ["northwind.example"])

    # --- Infrastructure -------------------------------------------------------
    redis_url: str | None = None  # set -> limits are shared by all pods (RedisLimiter)
    chroma_host: str | None = None  # set -> Chroma server over HTTP instead of a local directory
    chroma_port: int = 8000

    # --- Conversation history (PostgreSQL) ---------------------------------------
    database_url: str | None = None  # unset -> in-memory history (offline / tests)
    db_pool_min: int = 1
    db_pool_max: int = 10  # per pod; pods x max must stay below Postgres max_connections
    history_retention_days: int = 90  # purge CronJob deletes conversations idle this long
    # Save-first turns: the question is stored as 'processing' before the pipeline runs.
    turn_heartbeat_s: float = 10.0  # worker renews its claim this often while generating
    turn_stale_after_s: float = 90.0  # sweep CronJob fails 'processing' rows with no heartbeat this long
    turn_max_attempts: int = 3  # a failed question can be retried (same message_id) this many times

    # --- Concurrency & timeouts -----------------------------------------------
    # Limits are global when redis_url is set, per process otherwise.
    max_concurrent_chats: int = 50  # admission control for /chat and /chat/stream
    max_streams_per_user: int = 2
    generator_concurrency: int = 20  # simultaneous generation calls to the LLM provider
    guard_concurrency: int = 40  # classifier / moderation / grounding calls (short)
    slot_wait_s: float = 5.0  # wait this long for a slot, then 503 + Retry-After
    llm_timeout_s: float = 20.0  # guard-type calls
    generation_timeout_s: float = 60.0  # full streamed generation
    lease_ttl_s: float = 30.0  # Redis lease; renewed while held, frees slots of crashed pods

    # --- Live agent handoff ---------------------------------------------------
    handoff_on_no_answer: bool = True  # KB has no answer -> forward to a human
    handoff_on_unverified: bool = True  # answer failed verification twice -> forward
    require_user_id: bool = False  # True in production: conversations need an authenticated user
    groups_claim: str = "groups"  # JWT claim with the user's groups -> document access (access.py)
    agent_webhook_secret: str | None = None  # HMAC key shared with the desk; webhook off if unset
    conversation_ttl_s: int = 24 * 3600  # how long agent messages / conversation mode are kept

    @property
    def relevance_threshold(self) -> float:
        return self.offline_min_relevance if self.offline else self.min_relevance

    @property
    def rerank_threshold(self) -> float:
        return self.offline_rerank_min_score if self.offline else self.rerank_min_score

    @model_validator(mode="after")
    def _check_holdback(self) -> "Settings":
        from northwind_rag.guardrails.patterns import MAX_MATCH_CHARS

        if self.stream_holdback_chars < MAX_MATCH_CHARS:
            raise ValueError(
                f"stream_holdback_chars ({self.stream_holdback_chars}) must be >= "
                f"MAX_MATCH_CHARS ({MAX_MATCH_CHARS}) or split secrets can leak"
            )
        if self.reranker != "none" and self.rerank_candidates < self.top_k:
            raise ValueError(f"rerank_candidates ({self.rerank_candidates}) must be >= top_k ({self.top_k})")
        if self.turn_stale_after_s < 3 * self.turn_heartbeat_s:
            raise ValueError(
                f"turn_stale_after_s ({self.turn_stale_after_s}) must be >= 3 x turn_heartbeat_s "
                f"({self.turn_heartbeat_s}) or a slow heartbeat gets a live answer swept"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
