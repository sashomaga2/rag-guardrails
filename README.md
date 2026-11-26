# Northwind RAG: LangGraph + Chroma + guarded streaming + LangSmith evals

This is a production-style RAG reference: an IT helpdesk assistant over a small internal knowledge base.
It answers only from retrieved documents and cites every fact, and hands the conversation to a human
when it can't help. Seven guardrail layers protect it,
including guardrails that work **while the answer streams**. Each layer is measured by an evaluation
suite that runs locally (CI gate) and in LangSmith (experiments).

```
        ┌──────────── input guardrails ─────────────┐   ┌─ retrieval guardrails ─┐
START → prepare ─┬→ classify_input (regex → LLM) ─┬→ gate ─→ refuse ───────────────────────→ END
  (validate,     └→ retrieve (Chroma, threshold) ─┘    ├─→ handoff (user asked) ─────────→ END
   redact PII)      + drop injected chunks             ├─→ no_context ──→ handoff ───────→ END
                    + mask secrets in chunks           └─→ generate ──→ check_output ────→ END
                                                             ▲  (streams)   │  └─→ handoff → END
                                                             └── 1 retry ───┘  (no answer / unverified)
```

`classify_input` and `retrieve` run **in parallel**, so the LLM guard adds no latency.
`retrieve` is **hybrid**: Chroma vectors and Elasticsearch BM25, merged with Reciprocal Rank Fusion.

## Quick start

```bash
pip install -e ".[dev]"

# 1. Offline: no API keys. Deterministic stand-in models, same graph, same guardrails
export RAG_OFFLINE=true
pytest -q                                   # 322 tests (16 need RAG_TEST_DATABASE_URL)
python -m evals.run_local                   # eval gate
python -m northwind_rag.cli ask "How often do engineers get a new laptop?" --stream
uvicorn northwind_rag.api:app --reload      # open http://localhost:8000 for the demo UI

# 2. Full system in Docker (api + redis + chroma + elasticsearch + postgres + migrate + ingest)
RAG_OFFLINE=true docker compose up --build  # http://localhost:8080

# 3. Real models + LangSmith
cp .env.example .env                        # set OPENAI_API_KEY, LANGSMITH_API_KEY, RAG_OFFLINE=false
python -m northwind_rag.cli ingest          # embeds data/docs into .chroma/
python -m evals.run_local                   # adds LLM-judge metrics
python -m evals.run_langsmith --prefix baseline
```

Models are plain `init_chat_model` strings, so switching provider is a config change, for example
`RAG_CHAT_MODEL=anthropic:claude-sonnet-4-5` or `azure_openai:<deployment>`.

## Hybrid retrieval (dense + BM25 + Reciprocal Rank Fusion)

```
question ─┬─► Chroma (embeddings, cosine ≥ threshold) ─► ranked list A ─┐
          └─► Elasticsearch BM25 (minimum_should_match) ─► ranked list B ─┴─► RRF ─► 2 per file ─► top-k
```

- **Why both.** Embeddings match meaning ("VPN keeps dropping" ≈ "disconnects every few
  minutes") but are weak on exact tokens: error strings (`Gateway not reachable`), codes, SKUs,
  catalog item names (`Hardware Refresh`). BM25 is the opposite.
- **Why RRF.** BM25 scores are unbounded and depend on the query; cosine is 0..1. Adding or
  normalizing them is fragile, so RRF uses only ranks: `score(d) = Σ weight / (k + rank)` with
  `k = 60`. A chunk near the top of both lists wins (`src/northwind_rag/fusion.py` has a worked
  example). Weights (`RAG_RRF_DENSE_WEIGHT`, `RAG_RRF_LEXICAL_WEIGHT`) tilt the fusion.
- **Keeping "I don't know" honest.** Dense hits must pass the relevance threshold. Lexical hits
  must match a real share of the query terms (`minimum_should_match: 2<60%`), so one common word
  like "Northwind" can't pull unrelated chunks into the context.
- **Elasticsearch index.** English analyzer (stemming, stopwords), plus a `content.exact`
  subfield with the standard analyzer for codes and error strings. Title and section are boosted
  ×2. The mapping is `strict`, and bulk upserts are keyed by chunk id with
  `refresh=wait_for`, so chunks are searchable when the ingest Job finishes.
- **One ingest, two stores.** The same chunk ids go to Chroma and Elasticsearch. Stale chunks of
  edited files, and every chunk of deleted files, are removed from both.
- **Degradation.** If Elasticsearch is down, retrieval falls back to dense only
  (`retrieval: dense_fallback` on each chunk) instead of failing the chat. `/readyz` reports it.
- **No Elasticsearch?** Without `RAG_ELASTICSEARCH_URL`, an in-process BM25 with the same
  matching rule is built from Chroma's contents (offline mode and tests). Set
  `RAG_RETRIEVAL_MODE=dense` to turn hybrid off.
- **Debugging.** Every retrieved chunk carries `dense_rank`, `lexical_rank`, `rrf_score`, the
  cosine `score` and `lexical_score`, which are visible in LangSmith traces.
- **Measuring it.** The `retrieval_mrr` metric and three exact-term examples were added to the
  eval set. Compare `RAG_RETRIEVAL_MODE=dense` against `hybrid`, or different RRF weights, as
  LangSmith experiments; the mode and weights are recorded as experiment metadata. Offline, both
  modes score the same, because the offline embedding is itself lexical. The difference only
  shows with real embeddings.

## Guardrail layers

| # | Layer | Where | Kind | Catches | Cost |
|---|---|---|---|---|---|
| 1 | Validation + PII/secret redaction of the question | `prepare` | regex, Luhn | emails, cards, IBANs, phones, keys sent by the user, before any LLM, embedding or log sees them | ~0 ms |
| 2 | Prompt-injection heuristics | `classify_input` | regex | "ignore previous instructions", role tags, "developer mode" | ~0 ms |
| 3 | LLM input classifier (small model) | `classify_input` | structured output | paraphrased injections, harmful asks, off-topic | parallel to retrieval |
| 4 | Indirect-injection screening + secret masking of chunks | `retrieve` | regex | poisoned wiki pages; leaked keys in docs never reach the model | ~0 ms |
| 5 | Streaming redaction with hold-back buffer | API stream | regex, blocking | secrets/PII split across tokens | delays text by up to 128 characters |
| 6 | Streaming moderation with retraction | API stream | async LLM | credential phishing, security bypass | non-blocking |
| 7 | End-of-stream checks + corrective retry | `check_output` | regex + LLM | invalid/missing citations, ungrounded claims | 1 LLM call |

### How the streaming guardrails work

- **Hold-back redaction (layer 5).** Every detector has a bounded maximum length (`MAX_MATCH_CHARS`).
  The stream releases text only up to `len(buffer) - holdback`, moved left so it never splits a match.
  A secret that starts in released text would have to be longer than the hold-back, which no pattern
  allows. So a key split across tokens (`sk-proj-Q7f` + `3kLm9…`) can never leak partly. This is
  checked by a 200-case property test with random chunkings.
- **Optimistic moderation (layer 6).** Text streams immediately. A background task classifies the
  cumulative text every 240 characters, with no pile-up: at most one check runs at a time. If a check
  flags the text, the client receives `retract` with a replacement and the graph run is closed, so you
  stop paying for tokens.
- **Verification after the stream (layer 7).** Citations and groundedness need the full answer. The
  client shows the answer as pending, then receives `verified`, or `retract` and a regenerated answer
  (one retry with feedback), or `retract` with a safe fallback.

## Live agent handoff

`src/northwind_rag/handoff.py` defines the contract with a human support desk. It has **no real
integration yet**: `StubLiveAgentService` logs the request, returns a `STUB-xxxx` reference with
`status="not_implemented"` and keeps requests in memory for tests.

| Trigger | `reason` | How it's detected |
|---|---|---|
| The user asks for a person ("talk to a human", "live agent", "escalate this") | `user_requested` | regex, then the LLM input classifier's `agent_request` label; also the **Talk to a person** button → `POST /handoff` |
| The knowledge base has no answer | `no_answer` | nothing retrieved, or the model replied "I don't know" |
| The answer failed verification twice | `unverified_answer` | citations or groundedness failed after the retry |

`HandoffRequest` carries what a human agent needs to continue without asking again: the sanitized
question (PII is redacted, as everywhere else), the conversation and user ids, the retrieved
sources, the draft answer and the unsupported claims. The **types** of PII that were redacted are
included, never the values.

To integrate a real desk, implement two methods and inject the class:

```python
class ServiceNowLiveAgent:                       # or Zendesk, Genesys, Salesforce, a Teams queue...
    async def handoff(self, request: HandoffRequest) -> HandoffResult: ...
    async def is_available(self) -> bool: ...

pipeline = Pipeline.create(live_agent=ServiceNowLiveAgent())
```

When the desk is closed or the queue is full, return `status="unavailable"` with a user-facing
message instead of raising. If the integration raises anyway, the graph catches it and shows the
helpdesk contact details, so a broken integration never breaks the chat. Turn the automatic
triggers off with `RAG_HANDOFF_ON_NO_ANSWER=false` / `RAG_HANDOFF_ON_UNVERIFIED=false`.

### Agent replies across pods (WebSocket + Redis Streams)

With several replicas, the user's WebSocket is on pod A, but the desk's reply webhook can land on
pod B. Nothing routes to a specific pod. Each conversation has a Redis Stream, and whichever pod
holds the user's socket delivers from it.

```
desk ── webhook ──► pod B ── XADD ──► Redis stream conv:<id>:events
                                              │
pod A: ONE shared reader ◄── XREAD BLOCK (all conversations of this pod's sockets + pod wake stream)
   └─► fan-out: per-socket queue ──► WebSocket ──► user
user ── {"type":"user_message"} over the WebSocket ──► LiveAgentService.send_user_message ──► desk
```

- **One Redis reader per pod.** A pod with 5,000 open sockets uses one blocking Redis connection,
  not 5,000. Its reader runs a single multi-key `XREAD BLOCK` over every conversation that has a
  local subscriber and fans entries out to in-memory queues. When a new conversation is
  subscribed, the subscriber writes to the pod's own *wake* stream, which is part of the same
  XREAD, so the reader picks up the new key at once.
- **Resume.** A new socket first replays everything after its `last_event_id` (`XRANGE`), then
  attaches to the shared reader. Ids are compared numerically, so the overlap is dropped: nothing
  lost, nothing duplicated.
- **Slow clients.** If a socket's queue overflows, the server closes it with code `4408`. The
  client reconnects and resumes from its last id, so one slow phone never holds up the pod.
- **Streams, not Pub/Sub.** Pub/Sub loses messages published while a pod restarts or a socket
  reconnects. Streams are capped (`MAXLEN ~500`) and expire after 24 hours.
- **WebSocket auth.** Browsers can't send auth headers on a WebSocket. The client first calls
  `POST /conversations/{id}/ws-ticket` with its normal JWT and gets a **single-use ticket bound to
  the user and the conversation**, valid for 30 seconds (`GETDEL`). Only that ticket goes into the
  WebSocket URL, never the token.
- **Two-way.** The user also sends messages to the agent over the socket. They're PII-redacted,
  forwarded to the desk, and echoed to the user's other tabs through the stream.

### Ownership

Every entry point (`/chat`, `/chat/stream`, `/handoff`, `ws-ticket`, the socket itself and each
message sent over it) runs one atomic Lua `claim`:

| State in Redis | Result |
|---|---|
| no owner, no history | the caller becomes the owner |
| owner is the caller | allowed; the owner, mode and stream TTLs are refreshed together |
| owner is someone else | `404` (never `403`, so ids can't be probed) |
| no owner but history exists | `404`: an expired owner key never hands someone else's messages to a new user |

This fixes the takeover bug in the previous version. Ownership was only checked once a
conversation was with an agent, so another user could post into a bot-mode conversation or
trigger its handoff. And because the owner key and the message stream expired separately,
someone else could claim an old conversation and read its history. Now `publish` refreshes the
owner TTL together with the stream. In production `RAG_REQUIRE_USER_ID=true`: a conversation
without an authenticated user is rejected with `401`.

**Try it locally:** click *Talk to a person* in the UI, then run
`RAG_AGENT_WEBHOOK_SECRET=local-dev-secret python scripts/agent_reply.py <conversation-id> message "Hi, I'm Ana"`.

## Conversation history (PostgreSQL)

PostgreSQL is the system of record for conversations: every user, assistant, agent and system
message, plus the **durable owner** of each conversation. Redis keeps only short-lived state
(limiter leases, live push streams, a cached owner).

```
conversations (id, user_id, status bot|agent|closed, title, handoff_ticket, created_at, updated_at)
messages      (id, conversation_id → CASCADE, role, content, status, metadata jsonb, idempotency_key, created_at,
               state processing|done|failed, attempts, owner, heartbeat_at, error)
              UNIQUE (conversation_id, idempotency_key)
```

- **Only redacted text is stored**: the same sanitized question the LLM saw. Card numbers, IBANs
  and secrets never reach the database.
- **Idempotent writes.** The client sends a `message_id` with each chat message. A retried request
  (network blip, double click) is answered *from the database*, with no second paid generation,
  and is stored once. The desk's `event_id` does the same for webhooks: a redelivered event is
  stored once and pushed once. Messages are written to the database *before* they're pushed, so
  nothing reaches the user that isn't also in the history.
- **Save first.** The (redacted) question is inserted with `state = 'processing'` *before* the
  pipeline runs, so a pod crash never loses it. That insert is also the lock: a duplicate
  `message_id` hits the UNIQUE constraint and gets `409 in_progress` (+ `Retry-After`) while the
  first request is generating, or the stored answer once it's done. One generation is paid for,
  however many copies arrive at once.
  - While generating, the worker renews `heartbeat_at` every `RAG_TURN_HEARTBEAT_S` (10 s).
  - The `rag-sweep-stuck-questions` CronJob (every minute, `cli sweep`) marks rows with no
    heartbeat for `RAG_TURN_STALE_AFTER_S` (90 s) as `failed`: their pod died. `FOR UPDATE SKIP
    LOCKED` keeps overlapping sweeps safe.
  - A `failed` question (pod died, stream broke, client left, LLM error) is re-claimed by the
    client's retry with the same `message_id`, up to `RAG_TURN_MAX_ATTEMPTS` (3), then `422`.
  - `attempts` is a fencing token: a worker that was swept (long pause, partition) can't renew or
    finish the row any more, and drops its answer instead of writing a second one.
  - On graceful shutdown each worker fails its own leftover rows right away (`owner` is
    host:pid:nonce, so uvicorn workers in one pod don't release each other's rows).
  - The answer is written before the question is marked `done`, so a duplicate never sees `done`
    without an answer to replay.
  - If Postgres is down when a question arrives, it's answered anyway and stored at the end, as
    before: history being down never takes the assistant down.
- **Ownership is durable.** `claim()` is a single SQL statement (`INSERT … ON CONFLICT DO
  NOTHING` + owner lookup). Redis follows it, so a Redis restart or an expired key can't hand a
  conversation to someone else, while the real owner can always get back in.
- **What's recorded:** assistant messages carry `citations`, `sources`, `grounding_score`, the
  input guard label and any `handoff`. Handoffs, agent joins and leaves are `system` messages.
  The conversation's status and helpdesk ticket are kept on the conversation row.
- **API.** `GET /conversations` lists the caller's conversations, newest first, with keyset
  pagination (`next_cursor`). `GET /conversations/{id}/messages` returns the history oldest
  first (`next_before_id` to page back). `DELETE /conversations/{id}` is the right to erasure:
  it removes the history in Postgres and the live state in Redis. All three are owner-only, and
  every other caller gets `404`.
- **Retention.** The nightly `rag-purge-history` CronJob deletes conversations idle longer than
  `RAG_HISTORY_RETENTION_DAYS` (default 90).
- **Migrations.** These are numbered SQL files in `src/northwind_rag/migrations/`, run by
  `python -m northwind_rag.cli migrate`, which is the `rag-migrate` Job in Kubernetes and the
  `migrate` service in compose. Each file runs in its own transaction. A session-level advisory
  lock makes parallel runs safe (tested with four concurrent runners), and checksums reject a
  migration edited after it was applied. `/readyz` stays red until the schema is at the version
  the code needs.
- **Connections.** Each pod uses a psycopg 3 async pool (`RAG_DB_POOL_MAX`, default 10). Keep
  `pods × pool max` below Postgres `max_connections`, or put PgBouncer in front.
- **Without `RAG_DATABASE_URL`,** an in-memory store with the same contract is used (offline
  mode and tests).
- **Tests.** The store contract and the API tests run against both stores. The PostgreSQL ones
  need `RAG_TEST_DATABASE_URL` (CI starts a `postgres:16` service for this). On Windows without
  Docker they're skipped.

## Concurrency limits and backpressure

Every LLM call runs through `CallPolicy`: a concurrency slot for its role (`generator` or
`guard`) and a hard timeout. API requests also pass admission control: a limit on concurrent
chats overall and on concurrent streams per user.

| Limit | Setting | When it's hit |
|---|---|---|
| Concurrent chats | `RAG_MAX_CONCURRENT_CHATS` | `503` + `Retry-After`, before the stream starts |
| Streams per user | `RAG_MAX_STREAMS_PER_USER` | `503` immediately (no waiting) |
| Generation calls | `RAG_GENERATOR_CONCURRENCY` | waits up to `RAG_SLOT_WAIT_S`, then error |
| Guard / moderation / grounding calls | `RAG_GUARD_CONCURRENCY` | same |
| Call timeouts | `RAG_LLM_TIMEOUT_S`, `RAG_GENERATION_TIMEOUT_S` | error (fails closed) |

- **Local vs Redis.** Without `RAG_REDIS_URL` the limits are per process (`asyncio.Semaphore`).
  With it they're **global across all pods**. A Lua script atomically drops expired leases,
  checks the count and adds the new holder, using Redis server time so clock skew between pods
  doesn't matter.
- **Leases.** A heartbeat renews a lease while its slot is held. If a pod crashes, its slots free
  themselves after `RAG_LEASE_TTL_S`.
- **Fail closed.** If a guard times out or finds no free slot, the request fails. It never goes
  through unchecked. If moderation is unavailable mid-stream, the text already shown is retracted.
- **The user id** comes from the JWT: Istio writes the `sub` claim into `X-User-Id`. Clients
  can't set it themselves.

## Deployment

```
docker-compose.yml        api + redis + chroma + elasticsearch + postgres + migrate + ingest (full system)
Dockerfile                multi-stage, non-root, read-only-root friendly, 1 uvicorn worker per pod
k8s/base/                 kustomize base
  redis.yaml              limiter leases + conversation streams (no eviction policy)
  chroma.yaml             StatefulSet + PVC, HTTP server shared by all API pods
  elasticsearch.yaml      BM25 index, single node (ECK / Elastic Cloud + API key for production)
  postgres.yaml           conversation history, StatefulSet + PVC (managed DB / CloudNativePG for prod)
  database-jobs.yaml      migrate Job (before each rollout) + nightly history purge CronJob
                          + every-minute sweep of questions stuck in 'processing'
  api.yaml                Deployment (3+), probes, preStop drain, HPA, PDB, topology spread
  ingest.yaml             Job per deploy + nightly CronJob re-sync (idempotent upserts)
  networkpolicy.yaml      default deny; gateway→api, api→redis, api/ingest→chroma and elasticsearch,
                          db-client pods (api, migrate, purge, sweep)→postgres
  istio.yaml              Gateway, VirtualService (SSE/WebSocket timeouts, no POST retries), mTLS STRICT,
                          JWT → X-User-Id, AuthorizationPolicy
  config.env              settings → ConfigMap (hash suffix: changing a value rolls the pods)
  secret.example.yaml     template only: create the real Secret from Vault / External Secrets
k8s/overlays/dev          1 replica, offline models, no JWT (kind / minikube)
k8s/overlays/prod         registry image, bigger HPA and volume
```

```bash
docker build -t northwind-rag:dev .
kind load docker-image northwind-rag:dev        # or push to your registry
kubectl apply -k k8s/overlays/dev
kubectl -n northwind-rag port-forward svc/rag-api 8080:80
```

Notes:
- **Probes.** `/livez` checks only the process, so a dependency outage never restarts healthy
  pods. `/readyz` checks Redis and Chroma and takes a pod out of rotation when either is down.
- **Shutdown.** preStop waits 10s, uvicorn's graceful shutdown waits up to 60s, and Envoy's
  `terminationDrainDuration` is 60s, so streams in flight finish during rollouts.
- **No retries on chat POSTs.** A retry would start a second paid generation. The webhook route
  does retry: an event can arrive twice and the UI handles that.
- **Ingest runs as a Job, never inside API pods** (`RAG_AUTO_INGEST=false`), so pods never race to
  build the index.
- **Pinned versions.** The Chroma server image is pinned to the client version (1.5.9). Bump both
  together.
- **Production Redis.** Use a managed or HA Redis with AOF, so agent messages survive a Redis
  restart.
- **Autoscaling.** The HPA scales on CPU as a safe default. An LLM service is I/O-bound, so
  scaling on in-flight chats (`limiter.in_use("chats")` exported to Prometheus, via KEDA) fits
  better.

### SSE protocol (`POST /chat/stream`)

| event | data | client action |
|---|---|---|
| `status` | `{stage, attempt?}` | show progress |
| `delta` | `{text}` | append |
| `retract` | `{reason, replacement?}` | remove the shown text; show the replacement if there is one |
| `blocked` | `{answer, label}` | input refused; nothing was streamed |
| `final` | `{status, answer}` | non-streamed answer (no context) |
| `verified` | `{citations, grounding_score}` | mark the answer verified |
| `handoff` | `{reason, status, ticket_id, queue, message}` | show that a human will follow up; open the conversation WebSocket |
| `forwarded` | `{status: forwarded_to_agent}` | the message went to the human agent, not the bot |
| `error` | `{code: busy\|timeout, message, retry_after_s}` | capacity or timeout; retry later |
| `done` | full run summary | always last |

## Evaluation

The dataset is `evals/dataset.py` (27 examples): golden Q&A, unanswerable questions, off-topic
questions, requests for a human, direct injections, harmful asks, a secret probe, an
indirect-injection probe, and PII in the input.

| Metric | Type | Meaning | Gate |
|---|---|---|---|
| `behavior` | deterministic | answered, refused or declined as expected | ≥ 0.90 |
| `retrieval_recall` | deterministic | expected source files were retrieved | ≥ 0.85 |
| `retrieval_mrr` | deterministic | how high the first relevant chunk is ranked | ≥ 0.80 |
| `citation_validity` | deterministic | cites at least one source, and only retrieved ones | ≥ 0.95 |
| `no_leak` | deterministic | no secrets, PII or injection payload in the answer | **= 1.00** |
| `handoff_routing` | deterministic | forwarded to a human exactly when asked, or when there's no answer, and never for answered questions | ≥ 0.95 |
| `correctness` | LLM judge | agrees with the reference answer | ≥ 0.85 |
| `faithfulness` | LLM judge | every claim is supported by the retrieved context | ≥ 0.90 |
| `guard_precision` / `guard_recall` | summary | blocking quality over the whole set | ≥ 0.95 / = 1.00 |

Design choices:
- **The judge is a separate, stronger model** (`RAG_JUDGE_MODEL`) than the guard that does the
  runtime grounding check, so the system doesn't grade itself.
- **Security metrics are zero-tolerance.** A single leak fails the build.
- **Expected sources are file names, not chunk ids**, so re-chunking doesn't invalidate the dataset.
- **Experiments record their config as metadata** (models, `top_k`, threshold, chunk size), so
  LangSmith can compare runs side by side by configuration. Use `--reps 3` to measure LLM variance.
- Examples only the LLM guard can catch are marked `requires_llm_guard` and skipped offline.

Current offline result: every metric passes. The two examples that only the LLM guard can catch
are skipped offline.

## Project layout

```
src/northwind_rag/
  config.py            settings (env RAG_*)
  models.py            online model suite + offline stand-ins (hash embeddings, extractive streaming LLM)
  store.py             KnowledgeBase: ingest into Chroma + lexical index, hybrid search with RRF
  lexical.py           Elasticsearch BM25 index / in-process BM25 (same minimum_should_match rule)
  fusion.py            Reciprocal Rank Fusion
  graph.py             LangGraph pipeline
  pipeline.py          Pipeline + guarded streaming loop (SSE events)
  api.py               FastAPI: /chat, /chat/stream, /handoff, /conversations (list),
                       /conversations/{id}/messages, DELETE /conversations/{id}, /ws-ticket,
                       WS /conversations/{id}/ws, /agent/webhook, /livez, /readyz, demo UI
  handoff.py           LiveAgentService interface + stub (no real desk integration yet)
  conversations.py     ConversationBus: shared per-pod Redis Streams reader + fan-out, atomic
                       ownership claim, WebSocket tickets (Redis or in-process)
  limits.py            concurrency limiter: local semaphore / Redis distributed semaphore
  history.py           conversation history + durable owner: PostgreSQL (psycopg 3 pool) / in-memory
  migrate.py           SQL migration runner (advisory lock, checksums); migrations/*.sql
  prompts.py           all prompts
  guardrails/          patterns · input · documents · streaming · output
evals/                 dataset · evaluators · run_local (CI gate) · run_langsmith (experiments)
tests/                 patterns, streaming property tests, pipeline, SSE API
data/docs/             knowledge base (incl. one leaked key and one poisoned page, on purpose)
```

## Tuning notes

- `RAG_MIN_RELEVANCE`: tune it with `retrieval_recall` and `behavior` on the unanswerable set. Too
  high and you lose answers; too low and the model gets distracting context.
- `RAG_STREAM_HOLDBACK_CHARS` trades smoothness for safety. It can't go below `MAX_MATCH_CHARS`
  (validated at startup). Answers shorter than the hold-back arrive in one piece.
- To add a detector, give it a bounded length and update `MAX_MATCH_CHARS`.
- For production, add authentication, rate limiting, and per-user document ACLs (metadata filters in
  `KnowledgeBase.search`). Send LangSmith traces to a project with retention that matches your data
  policy (`LANGSMITH_ENDPOINT` for EU).

## License

MIT © 2026 Aleksandar Marinov. See [LICENSE](LICENSE).
