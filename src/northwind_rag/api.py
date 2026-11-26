"""FastAPI service: chat (JSON + SSE), live-agent WebSocket, conversation history, demo UI.

Run:  uvicorn northwind_rag.api:app --reload
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import hashlib
import hmac
import json
import logging
import os
import secrets
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from typing import Literal

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel, Field

from northwind_rag.access import parse_groups
from northwind_rag.guardrails.input import prepare_input
from northwind_rag.handoff import HandoffRequest
from northwind_rag.history import TurnClaim
from northwind_rag.limits import CapacityExceeded
from northwind_rag.conversations import SubscriberTooSlow
from northwind_rag.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)
ANONYMOUS = "anonymous"
# Owner of the question rows this process is answering. Per process, not per pod: several
# uvicorn workers in one pod must not release each other's rows on shutdown.
WORKER_ID = f"{socket.gethostname()}:{os.getpid()}:{secrets.token_hex(3)}"


MessageId = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)  # hard cap; the guard enforces the policy limit
    conversation_id: str | None = Field(default=None, max_length=128)
    # client-generated id: a retried request (network blip, double click) is answered from the
    # database instead of paying for a second generation, and is stored once
    message_id: str | None = MessageId


class AgentWebhook(BaseModel):
    """What the support desk sends when something happens in a handed-off conversation."""

    conversation_id: str = Field(min_length=1, max_length=128)
    type: Literal["agent_joined", "agent_message", "agent_left"]
    text: str = Field(default="", max_length=8000)
    agent_name: str | None = Field(default=None, max_length=100)
    event_id: str | None = MessageId  # desk's own event id: redelivered webhooks are applied once


class HandoffBody(BaseModel):
    """'Talk to a person' button: forward without going through the assistant."""

    question: str = Field(default="", max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=128)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = Pipeline.create()
    if (reranker := app.state.pipeline.kb.reranker) is not None:
        await reranker.warm_up()  # load the cross-encoder before the pod reports ready
    yield
    # Graceful shutdown: uvicorn has already drained in-flight requests; anything still marked
    # 'processing' by this worker is failed now, so a retry doesn't wait for the sweep CronJob.
    try:
        if released := await app.state.pipeline.history.release_owner(WORKER_ID):
            log.warning("released %d unfinished question(s) on shutdown", released)
    except Exception:
        log.exception("could not release unfinished questions; the sweep job will")
    await app.state.pipeline.bus.close()  # stop this pod's shared Redis reader
    await app.state.pipeline.history.close()  # return Postgres connections


app = FastAPI(title="Northwind IT Helpdesk RAG", lifespan=lifespan)


def _busy(retry_after_s: int = 2, code: str = "busy") -> JSONResponse:
    return JSONResponse(
        {"error": code, "message": "The assistant is busy right now. Please try again in a moment."},
        status_code=503,
        headers={"Retry-After": str(retry_after_s)},
    )


# In production the user id comes from the verified JWT: Istio RequestAuthentication copies the
# "sub" claim into this header (see k8s/istio.yaml), and the mesh strips it from client requests.
UserId = Header(default=None, alias="X-User-Id", max_length=128)
# Istio forwards the payload of the token it just verified (outputPayloadToHeader); the gateway
# strips any client-supplied copy first (k8s/istio.yaml). Its groups claim decides which documents
# retrieval may read. No header / no claim -> public documents only.
JwtPayload = Header(default=None, alias="X-Jwt-Payload", max_length=16384)


def user_groups(jwt_payload: str | None, claim: str = "groups") -> frozenset[str]:
    """base64url(JSON) -> the groups claim, as a list or a space/comma separated string.
    Anything malformed yields no groups: it can only ever narrow access, never widen it."""
    if not jwt_payload:
        return frozenset()
    try:
        raw = base64.urlsafe_b64decode(jwt_payload + "=" * (-len(jwt_payload) % 4))
        claims = json.loads(raw)
        value = claims.get(claim) if isinstance(claims, dict) else None
        return parse_groups(value if isinstance(value, (str, list)) else None, strict=False)
    except (binascii.Error, ValueError, UnicodeDecodeError):
        log.warning("ignoring malformed X-Jwt-Payload header")
        return frozenset()


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/livez")
async def livez() -> dict:
    """Liveness: the process is up. Never checks dependencies (that would restart healthy pods)."""
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(request: Request):
    """Readiness: Redis and Chroma reachable. Failing pods are taken out of the Service."""
    p: Pipeline = request.app.state.pipeline
    checks = await p.ready()
    body = {"status": "ok" if all(checks.values()) else "degraded", "checks": checks, "offline": p.settings.offline}
    return JSONResponse(body, status_code=200 if all(checks.values()) else 503)


@app.get("/health")
async def health(request: Request) -> dict:
    p: Pipeline = request.app.state.pipeline
    return {"status": "ok", "offline": p.settings.offline, "chunks": p.kb.count()}


# ------------------------------------------------------------------------ ownership gate
async def authorize_conversation(p: Pipeline, conversation_id: str | None, user_id: str | None) -> str | None:
    """Single gate for every conversation operation (chat, stream, handoff, WebSocket ticket).

    Claims the conversation for the caller on first use; anyone else gets 404 afterwards.
    Returns the effective user id. 404 (not 403) so ids can't be probed.
    """
    if not conversation_id:
        return user_id
    if not user_id:
        if p.settings.require_user_id:
            raise HTTPException(status_code=401, detail="authentication required")
        user_id = ANONYMOUS  # local/dev only: the random conversation id is the only secret
    # 1. PostgreSQL is the durable owner record (survives Redis restarts and TTLs)
    if not await p.history.claim(conversation_id, user_id):
        raise HTTPException(status_code=404)
    # 2. Redis caches it for the hot paths (WebSocket, per-message checks). `verified` lets the
    #    real owner re-claim after the Redis key expired even though the stream still exists.
    if not await p.bus.claim(conversation_id, user_id, verified=True):
        raise HTTPException(status_code=404)
    return user_id


async def require_owner(p: Pipeline, conversation_id: str, user_id: str | None) -> str:
    """Read/delete endpoints: never create a conversation, only check the durable owner."""
    user_id = user_id or (None if p.settings.require_user_id else ANONYMOUS)
    if user_id is None:
        raise HTTPException(status_code=401, detail="authentication required")
    if await p.history.owner(conversation_id) != user_id:
        raise HTTPException(status_code=404)
    return user_id


# ------------------------------------------------------------------------- persistence
def _key(prefix: str, message_id: str | None) -> str | None:
    return f"{prefix}:{message_id}" if message_id else None


class Turn:
    """The user's question row, saved as 'processing' before the pipeline runs.

    While the answer is generated, `start()` renews the heartbeat every `turn_heartbeat_s`. If
    this pod dies, the heartbeat stops and the sweep CronJob marks the row 'failed', so the
    client's retry (same message_id) can claim it again. The heartbeat starts only once work
    really starts: a response that is never streamed must not keep its row alive forever.
    """

    def __init__(self, p: Pipeline, claim: TurnClaim) -> None:
        self.history, self.claim, self.every = p.history, claim, p.settings.turn_heartbeat_s
        self.lost = False  # the sweeper took the row: this worker must not write the answer
        self._task: asyncio.Task | None = None

    @property
    def _ids(self) -> tuple[int, str, int]:
        return self.claim.message_id, WORKER_ID, self.claim.attempt

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._beat())

    async def _beat(self) -> None:
        while True:
            await asyncio.sleep(self.every)
            try:
                if not await self.history.heartbeat(*self._ids):
                    self.lost = True
                    log.warning("question %s was taken over (no heartbeat in time)", self.claim.message_id)
                    return
            except Exception:  # database blip: keep trying, the stale window is several beats long
                log.warning("heartbeat for question %s failed", self.claim.message_id, exc_info=True)

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()

    async def still_owner(self) -> bool:
        return not self.lost and await self.history.heartbeat(*self._ids)

    async def finish(self, *, failed: bool = False, error: str | None = None) -> bool:
        self.stop()
        try:
            return await self.history.finish_turn(*self._ids, failed=failed, error=error)
        except Exception:
            log.exception("could not finish question %s; the sweep job will", self.claim.message_id)
            return False


async def begin_turn(p: Pipeline, body: ChatRequest) -> Turn | TurnClaim | None:
    """Save the (redacted) question BEFORE generating. Returns a Turn this worker owns, the
    TurnClaim of a duplicate (processing / done / gave_up), or None when the question isn't
    tracked (no conversation, invalid input, or the database is down: then it's stored at the
    end as before, so an outage of history never takes the assistant down)."""
    if not body.conversation_id:
        return None
    prepared = prepare_input(body.question, p.settings)
    if not prepared.valid:
        return None  # the guard answers it right away; nothing to protect
    try:
        claim = await p.history.begin_turn(
            body.conversation_id, prepared.sanitized, owner=WORKER_ID,
            metadata={"pii_redacted": prepared.pii_found}, idempotency_key=_key("user", body.message_id),
            max_attempts=p.settings.turn_max_attempts,
        )
    except Exception:
        log.exception("could not save question first for conversation %s", body.conversation_id)
        return None
    return Turn(p, claim) if claim.outcome == "claimed" else claim


async def duplicate_response(p: Pipeline, body: ChatRequest, claim: TurnClaim) -> dict | JSONResponse:
    if claim.outcome == "processing":  # same message_id is being answered right now (double click, retry)
        return JSONResponse({"error": "in_progress", "message": "This message is still being answered."},
                            status_code=409, headers={"Retry-After": "2"})
    if claim.outcome == "gave_up":
        return JSONResponse({"error": "failed", "message": "This message could not be answered. Please send it again."},
                            status_code=422)
    # done: answered in the meantime, or a message the agent already received
    return await replay_if_duplicate(p, body) or {"answer": "", "status": "duplicate", "replayed": True}


async def record_turn(p: Pipeline, conversation_id: str | None, message_id: str | None, result: dict,
                      turn: Turn | None = None) -> None:
    """Store the final answer (and the question, unless it was saved first). Best effort: a history
    write failure is logged and alerts, but never takes away an answer the user already received.
    A row left 'processing' by a failure here is failed by the sweep job; its retry is replayed
    from the stored answer, because the answer is written before the question is marked done."""
    if not conversation_id:
        return
    try:
        if turn is not None:
            if not await turn.still_owner():  # swept as dead (and maybe retried): don't write a 2nd answer
                log.warning("dropping answer for question %s: no longer the owner", turn.claim.message_id)
                return
        elif result.get("question"):
            await p.history.append(conversation_id, "user", result["question"],
                                   metadata={"pii_redacted": result.get("input_pii_redacted", [])},
                                   idempotency_key=_key("user", message_id))
        meta = {k: result.get(k) for k in ("citations", "sources", "grounding_score", "unsupported_claims",
                                           "input_label", "attempts", "handoff") if result.get(k) not in (None, [], {})}
        await p.history.append(conversation_id, "assistant", result.get("answer", ""), status=result.get("status"),
                               metadata=meta, idempotency_key=_key("assistant", message_id))
        if turn is not None:
            await turn.finish()
        handoff = result.get("handoff")
        if handoff and handoff.get("status") != "unavailable":
            await p.history.set_status(conversation_id, "agent", handoff.get("ticket_id"))
    except Exception:
        log.exception("failed to persist turn for conversation %s", conversation_id)
    finally:
        if turn is not None:
            turn.stop()


async def replay_if_duplicate(p: Pipeline, body: ChatRequest) -> dict | None:
    """Same message_id again -> return the stored answer, generate nothing."""
    if not (body.conversation_id and body.message_id):
        return None
    stored = await p.history.find(body.conversation_id, _key("assistant", body.message_id))
    if stored is None:
        return None
    return {"answer": stored.content, "status": stored.status, "replayed": True, **stored.metadata}


async def forward_to_agent(p: Pipeline, conversation_id: str, text: str, message_id: str | None = None) -> dict:
    """While a human owns the conversation, user messages go to the desk, not to the bot."""
    prepared = prepare_input(text, p.settings)  # the desk gets redacted text too
    stored_id = await p.history.append(conversation_id, "user", prepared.sanitized, status="forwarded_to_agent",
                                       metadata={"to": "agent"}, idempotency_key=_key("user", message_id))
    if stored_id is None:  # retry of a message the agent already has
        return {"status": "forwarded_to_agent", "answer": "", "conversation_id": conversation_id, "replayed": True}
    await p.live_agent.send_user_message(conversation_id, prepared.sanitized)
    event_id = await p.bus.publish(conversation_id, "user_message", {"text": prepared.sanitized})
    return {"status": "forwarded_to_agent", "answer": "", "conversation_id": conversation_id, "event_id": event_id}


async def _agent_owned(p: Pipeline, conversation_id: str | None) -> bool:
    return bool(conversation_id) and await p.bus.get_mode(conversation_id) == "agent"


@app.post("/chat")
async def chat(body: ChatRequest, request: Request, user_id: str | None = UserId,
               jwt_payload: str | None = JwtPayload):
    p: Pipeline = request.app.state.pipeline
    groups = user_groups(jwt_payload, p.settings.groups_claim)
    user_id = await authorize_conversation(p, body.conversation_id, user_id)
    if await _agent_owned(p, body.conversation_id):
        return await forward_to_agent(p, body.conversation_id, body.question, body.message_id)
    if replayed := await replay_if_duplicate(p, body):
        return replayed
    try:
        async with p.admit(user_id):  # admit first: a 503 must not store (or burn an attempt of) the question
            turn = await begin_turn(p, body)
            if isinstance(turn, TurnClaim):
                return await duplicate_response(p, body, turn)
            if turn is not None:
                turn.start()
            try:
                result = await p.ask(
                    body.question, conversation_id=body.conversation_id, user_id=user_id, groups=groups,
                    run_name="chat", tags=["api"]
                )
            except BaseException as exc:  # incl. cancellation (client left): free it for a retry now
                if turn is not None:
                    await turn.finish(failed=True, error=repr(exc))
                raise
    except CapacityExceeded as exc:
        return _busy(exc.retry_after_s)
    except TimeoutError:
        return _busy(code="timeout")
    await record_turn(p, body.conversation_id, body.message_id, result, turn)
    # contexts are for evaluation/debugging, not for clients
    return {k: v for k, v in result.items() if k != "contexts"}


@app.post("/chat/stream")
async def chat_stream(body: ChatRequest, request: Request, user_id: str | None = UserId,
                      jwt_payload: str | None = JwtPayload):
    p: Pipeline = request.app.state.pipeline
    groups = user_groups(jwt_payload, p.settings.groups_claim)
    user_id = await authorize_conversation(p, body.conversation_id, user_id)
    if await _agent_owned(p, body.conversation_id):
        forwarded = await forward_to_agent(p, body.conversation_id, body.question, body.message_id)
        return StreamingResponse(
            iter([_sse("forwarded", forwarded), _sse("done", forwarded)]), media_type="text/event-stream"
        )
    if replayed := await replay_if_duplicate(p, body):
        return StreamingResponse(iter([_sse("final", replayed), _sse("done", replayed)]), media_type="text/event-stream")
    # Admit BEFORE the response starts, so overload is a clean 503 rather than a broken stream.
    admission = contextlib.AsyncExitStack()
    try:
        await admission.enter_async_context(p.admit(user_id))
    except CapacityExceeded as exc:
        return _busy(exc.retry_after_s)
    try:
        turn = await begin_turn(p, body)  # the question is in Postgres before anything is generated
    except BaseException:
        await admission.aclose()
        raise
    if isinstance(turn, TurnClaim):
        await admission.aclose()
        dup = await duplicate_response(p, body, turn)
        if isinstance(dup, Response):
            return dup
        return StreamingResponse(iter([_sse("final", dup), _sse("done", dup)]), media_type="text/event-stream")

    async def events() -> AsyncIterator[str]:
        summary: dict | None = None
        error = "client disconnected"
        if turn is not None:
            turn.start()
        try:
            async for ev in p.stream(
                body.question,
                conversation_id=body.conversation_id,
                user_id=user_id,
                groups=groups,
                run_name="chat_stream",
                tags=["api", "stream"],
            ):
                if await request.is_disconnected():
                    break  # client left: stop generating
                data = ev["data"]
                if ev["event"] == "done":
                    summary = data
                    data = {k: v for k, v in data.items() if k != "contexts"}
                yield _sse(ev["event"], data)
        except Exception as exc:
            error = repr(exc)
            raise
        finally:
            await admission.aclose()  # idempotent: the background task below is only a fallback
            if summary is not None:
                await record_turn(p, body.conversation_id, body.message_id, summary, turn)
            elif turn is not None:  # client left or the stream broke: 'failed', so a retry can run
                await turn.finish(failed=True, error=error)
            elif body.conversation_id:  # untracked question (database was down at the start): keep it
                prepared = prepare_input(body.question, p.settings)
                with contextlib.suppress(Exception):
                    await p.history.append(body.conversation_id, "user", prepared.sanitized,
                                           metadata={"aborted": True}, idempotency_key=_key("user", body.message_id))

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        background=BackgroundTask(admission.aclose),
    )


@app.post("/handoff")
async def handoff(body: HandoffBody, request: Request, user_id: str | None = UserId) -> dict:
    p: Pipeline = request.app.state.pipeline
    user_id = await authorize_conversation(p, body.conversation_id, user_id)
    prepared = prepare_input(body.question, p.settings)  # same PII redaction as the chat path
    result = await p.live_agent.handoff(
        HandoffRequest(
            reason="user_requested",
            question=prepared.sanitized,
            conversation_id=body.conversation_id,
            user_id=user_id,
            input_pii_redacted=prepared.pii_found,
        )
    )
    if body.conversation_id and result.status != "unavailable":
        await p.history.append(body.conversation_id, "system", "Conversation handed off to the IT helpdesk.",
                               status="handed_off", metadata={"handoff": result.to_dict(), "reason": "user_requested"})
        await p.history.set_status(body.conversation_id, "agent", result.ticket_id)
        await p.bus.set_mode(body.conversation_id, "agent")
        await p.bus.publish(body.conversation_id, "status", {"state": "waiting_for_agent", **result.to_dict()})
    return result.to_dict()


# ------------------------------------------------------------------------ history API
@app.get("/conversations")
async def list_conversations(request: Request, user_id: str | None = UserId,
                             cursor: str | None = Query(default=None, max_length=200),
                             limit: int = Query(default=20, ge=1, le=100)) -> dict:
    """The caller's conversations, most recent first. Pass `next_cursor` back for the next page."""
    p: Pipeline = request.app.state.pipeline
    user_id = user_id or (None if p.settings.require_user_id else ANONYMOUS)
    if user_id is None:
        raise HTTPException(status_code=401, detail="authentication required")
    try:
        page, next_cursor = await p.history.conversations(user_id, cursor=cursor, limit=limit)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid cursor") from None
    return {"conversations": [c.to_dict() for c in page], "next_cursor": next_cursor}


@app.get("/conversations/{conversation_id}/messages")
async def conversation_messages(conversation_id: str, request: Request, user_id: str | None = UserId,
                                before_id: int | None = Query(default=None, ge=1),
                                limit: int = Query(default=50, ge=1, le=200)) -> dict:
    """Full history (bot, agent and system messages), oldest first. Page back with before_id."""
    p: Pipeline = request.app.state.pipeline
    await require_owner(p, conversation_id, user_id)
    msgs = await p.history.messages(conversation_id, before_id=before_id, limit=limit)
    return {"messages": [m.to_dict() for m in msgs],
            "next_before_id": msgs[0].id if len(msgs) == limit else None}


@app.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str, request: Request, user_id: str | None = UserId) -> Response:
    """Right to erasure: removes the history (Postgres, cascades to messages) and live state (Redis)."""
    p: Pipeline = request.app.state.pipeline
    await require_owner(p, conversation_id, user_id)
    await p.history.delete(conversation_id)
    await p.bus.forget(conversation_id)
    return Response(status_code=204)


# ----------------------------------------------------------- live agent push: WebSocket
KEEPALIVE_S = 15.0  # app-level ping; below Envoy / load balancer idle timeouts
TICKET_TTL_S = 30
WS_CLOSE_UNAUTHORIZED = 4401
WS_CLOSE_TOO_SLOW = 4408  # client should reconnect with its last event id


@app.post("/conversations/{conversation_id}/ws-ticket")
async def ws_ticket(conversation_id: str, request: Request, user_id: str | None = UserId) -> dict:
    """Browsers can't send auth headers on a WebSocket. The client calls this with its normal
    (JWT-authenticated) request and gets a single-use ticket valid for 30 seconds, bound to the
    user and the conversation. The ticket, not a token, goes in the WebSocket URL."""
    p: Pipeline = request.app.state.pipeline
    user_id = await authorize_conversation(p, conversation_id, user_id)
    ticket = await p.bus.issue_ticket(conversation_id, user_id, TICKET_TTL_S)
    return {"ticket": ticket, "expires_in": TICKET_TTL_S}


@app.websocket("/conversations/{conversation_id}/ws")
async def conversation_ws(
    websocket: WebSocket,
    conversation_id: str,
    ticket: str = Query(max_length=64),
    last_event_id: str | None = Query(default=None, max_length=64),
) -> None:
    """Server -> client: {"type": "event", "event": {id, type, data}} | {"type": "ping"}
    Client -> server: {"type": "user_message", "text": "..."} | {"type": "pong"}

    Runs on whichever pod the user is connected to; the pod's shared Redis reader feeds it.
    """
    p: Pipeline = websocket.app.state.pipeline
    user_id = await p.bus.redeem_ticket(ticket, conversation_id)
    if user_id is None or not await p.bus.claim(conversation_id, user_id):
        await websocket.close(code=WS_CLOSE_UNAUTHORIZED)  # before accept -> handshake rejected
        return
    await websocket.accept()

    async def push() -> None:
        try:
            async with contextlib.aclosing(p.bus.subscribe(conversation_id, last_event_id, KEEPALIVE_S)) as events:
                async for ev in events:
                    await websocket.send_json({"type": "ping"} if ev is None else {"type": "event", "event": ev.to_dict()})
        except SubscriberTooSlow:
            await websocket.close(code=WS_CLOSE_TOO_SLOW)

    async def receive() -> None:
        while True:
            msg = await websocket.receive_json()
            if msg.get("type") != "user_message":
                continue  # pong / unknown: ignore
            text = str(msg.get("text", ""))[:4000]
            if not text.strip():
                continue
            if not await p.bus.claim(conversation_id, user_id):  # re-checked on every message
                await websocket.close(code=WS_CLOSE_UNAUTHORIZED)
                return
            if await _agent_owned(p, conversation_id):
                mid = msg.get("message_id")
                mid = mid if isinstance(mid, str) and 0 < len(mid) <= 64 else None
                await forward_to_agent(p, conversation_id, text, mid)  # echoed back to all tabs via the bus
            else:
                await websocket.send_json({"type": "error", "code": "not_with_agent",
                                           "message": "Send this through the chat; no agent is connected."})

    tasks = [asyncio.create_task(push()), asyncio.create_task(receive())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            if t.exception() and not isinstance(t.exception(), WebSocketDisconnect):
                log.warning("websocket %s ended with error", conversation_id, exc_info=t.exception())
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def verify_signature(secret: str, body: bytes, signature: str | None) -> bool:
    """Desk signs the raw body: X-Signature: sha256=<hex hmac>. Constant-time compare."""
    if not signature or not signature.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature.removeprefix("sha256="))


@app.post("/agent/webhook", status_code=202)
async def agent_webhook(request: Request, x_signature: str | None = Header(default=None)) -> dict:
    """Called by the support desk. Any pod can receive it; the bus delivers to the right one."""
    p: Pipeline = request.app.state.pipeline
    secret = p.settings.agent_webhook_secret
    if not secret:
        raise HTTPException(status_code=503, detail="agent webhook not configured")
    raw = await request.body()
    if not verify_signature(secret, raw, x_signature):
        raise HTTPException(status_code=401, detail="bad signature")
    event = AgentWebhook.model_validate_json(raw)
    if not await _agent_owned(p, event.conversation_id):  # never create streams for unknown ids
        raise HTTPException(status_code=409, detail="conversation is not with an agent")
    data = {"text": event.text, "agent_name": event.agent_name}
    # Store first, keyed by the desk's event id: a redelivered webhook stops here, so the user never
    # sees the same agent message twice. If the database is down this fails with 5xx and the
    # desk retries: nothing is pushed that isn't also in the history.
    role, content = ("agent", event.text) if event.type == "agent_message" else (
        "system", f"{event.agent_name or 'The agent'} {'joined' if event.type == 'agent_joined' else 'left'} the conversation.")
    stored = await p.history.append(event.conversation_id, role, content, status=event.type,
                                    metadata={"agent_name": event.agent_name}, idempotency_key=_key("desk", event.event_id))
    if stored is None:
        return {"accepted": True, "duplicate": True}
    event_id = await p.bus.publish(event.conversation_id, event.type, data)
    if event.type == "agent_left":  # the bot takes the conversation back
        await p.bus.set_mode(event.conversation_id, "bot")
        await p.history.set_status(event.conversation_id, "bot")
    return {"accepted": True, "event_id": event_id}


@app.get("/")
async def index(request: Request) -> FileResponse:
    return FileResponse(Path(request.app.state.pipeline.settings.static_dir) / "index.html")
