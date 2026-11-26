"""Live-agent messaging: shared per-pod Redis reader, WebSocket push, ownership."""

import asyncio
import contextlib
import dataclasses
import hashlib
import hmac
import json

import fakeredis
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from northwind_rag.api import app
from northwind_rag.conversations import LocalBus, RedisStreamBus, SubscriberTooSlow
from northwind_rag.graph import build_graph

SECRET = "test-webhook-secret"


@pytest.fixture
def redis_server():
    return fakeredis.FakeServer()


def redis_bus(server, **kw) -> RedisStreamBus:
    """One pod: its own command connection and its own reader connection, shared Redis server."""
    return RedisStreamBus(fakeredis.FakeAsyncRedis(server=server), fakeredis.FakeAsyncRedis(server=server), **kw)


async def take(agen, n, timeout=2.0):
    out = []

    async def run():
        async for item in agen:
            if item is not None:
                out.append(item)
                if len(out) == n:
                    return

    await asyncio.wait_for(run(), timeout)
    return out


# ================================================================================ bus level
@pytest.mark.parametrize("kind", ["local", "redis"])
async def test_backlog_then_live_events(kind, redis_server):
    publisher = LocalBus() if kind == "local" else redis_bus(redis_server)
    subscriber = publisher if kind == "local" else redis_bus(redis_server)  # other pod
    await publisher.publish("c1", "status", {"state": "waiting_for_agent"})
    sub = subscriber.subscribe("c1", keepalive_s=0.05)
    first = await take(sub, 1)
    later = asyncio.create_task(take(sub, 1))
    await asyncio.sleep(0.05)
    await publisher.publish("c1", "agent_message", {"text": "Hi, I'm Ana from IT."})
    assert [e.type for e in first + await later] == ["status", "agent_message"]
    await subscriber.close()


@pytest.mark.parametrize("kind", ["local", "redis"])
async def test_resume_after_last_event_id(kind, redis_server):
    bus = LocalBus() if kind == "local" else redis_bus(redis_server)
    ids = [await bus.publish("c2", "agent_message", {"text": f"m{i}"}) for i in range(3)]
    resumed = await take(bus.subscribe("c2", last_id=ids[0], keepalive_s=0.05), 2)
    assert [e.data["text"] for e in resumed] == ["m1", "m2"]  # nothing lost, nothing repeated
    await bus.close()


async def test_idle_subscription_yields_ping(redis_server):
    bus = redis_bus(redis_server, block_ms=50)
    assert await asyncio.wait_for(bus.subscribe("quiet", keepalive_s=0.05).__anext__(), 1) is None
    await bus.close()


async def test_one_shared_reader_per_pod(redis_server):
    """30 open connections over 10 conversations -> one blocking XREAD at a time, correct fan-out."""
    pod_a, pod_b = redis_bus(redis_server), redis_bus(redis_server)
    in_flight = peak = 0
    real_xread = pod_a.reader_redis.xread

    async def counting_xread(*args, **kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            return await real_xread(*args, **kwargs)
        finally:
            in_flight -= 1

    pod_a.reader_redis.xread = counting_xread
    subs = [(f"conv-{i % 10}", pod_a.subscribe(f"conv-{i % 10}", keepalive_s=1)) for i in range(30)]
    readers = [asyncio.create_task(take(s, 2, timeout=3)) for _, s in subs]
    await asyncio.sleep(0.1)
    for c in range(10):  # replies land on the other pod
        await pod_b.publish(f"conv-{c}", "agent_message", {"text": f"for conv-{c} #1"})
        await pod_b.publish(f"conv-{c}", "agent_message", {"text": f"for conv-{c} #2"})
    results = await asyncio.gather(*readers)
    for (cid, _), events in zip(subs, results):
        assert [e.data["text"] for e in events] == [f"for {cid} #1", f"for {cid} #2"]
    assert peak == 1  # a single Redis connection serves every socket of the pod
    await pod_a.close()


async def test_new_subscription_wakes_blocked_reader(redis_server):
    pod = redis_bus(redis_server, block_ms=5000)
    first = pod.subscribe("old", keepalive_s=10)
    first_task = asyncio.create_task(take(first, 1, timeout=10))
    await asyncio.sleep(0.1)  # reader is now blocked for up to 5s on "old" only
    second = pod.subscribe("new", keepalive_s=10)
    second_task = asyncio.create_task(take(second, 1, timeout=1.0))
    await asyncio.sleep(0.05)
    await redis_bus(redis_server).publish("new", "agent_message", {"text": "fast"})
    assert (await second_task)[0].data["text"] == "fast"  # well under the 5s block
    first_task.cancel()
    await pod.close()


@pytest.mark.parametrize("kind", ["local", "redis"])
async def test_slow_consumer_is_cut_off(kind, redis_server):
    bus = LocalBus(queue_size=2) if kind == "local" else redis_bus(redis_server, queue_size=2)
    sub = bus.subscribe("busy", keepalive_s=1)
    started = asyncio.create_task(sub.__anext__())  # attach, then stop reading
    await bus.publish("busy", "agent_message", {"text": "0"})
    await started
    await asyncio.sleep(0.05)
    for i in range(1, 6):
        await bus.publish("busy", "agent_message", {"text": str(i)})
    await asyncio.sleep(0.1)
    with pytest.raises(SubscriberTooSlow):
        async for _ in sub:
            pass
    await bus.close()


@pytest.mark.parametrize("kind", ["local", "redis"])
async def test_claim_rules(kind, redis_server):
    bus = LocalBus() if kind == "local" else redis_bus(redis_server)
    assert await bus.claim("c3", "alice")
    assert await bus.claim("c3", "alice")  # owner again: fine
    assert not await bus.claim("c3", "mallory")  # someone else: denied


async def test_orphaned_history_cannot_be_claimed(redis_server):
    bus = redis_bus(redis_server)
    assert await bus.claim("c4", "alice")
    await bus.publish("c4", "agent_message", {"text": "your laptop serial is ..."})
    await bus.redis.delete(bus._owner_key("c4"))  # owner key expired/evicted, history still there
    assert not await bus.claim("c4", "mallory")


@pytest.mark.parametrize("kind", ["local", "redis"])
async def test_tickets_single_use_and_bound(kind, redis_server):
    bus = LocalBus() if kind == "local" else redis_bus(redis_server)
    t = await bus.issue_ticket("c5", "alice")
    assert await bus.redeem_ticket(t, "other-conv") is None  # bound to its conversation (and burnt)
    t = await bus.issue_ticket("c5", "alice")
    assert await bus.redeem_ticket(t, "c5") == "alice"
    assert await bus.redeem_ticket(t, "c5") is None  # single use


# ================================================================================ API level
@contextlib.asynccontextmanager
async def _no_lifespan(_app):
    yield


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app.router, "lifespan_context", _no_lifespan)
    with TestClient(app) as c:  # one event loop for every call and socket in the test
        yield c


def pod(pipeline, server, **overrides):
    settings = pipeline.settings.model_copy(update={"agent_webhook_secret": SECRET, **overrides})
    bus = redis_bus(server, block_ms=200)
    graph = build_graph(pipeline.suite, pipeline.kb, settings, pipeline.live_agent, bus)
    return dataclasses.replace(pipeline, settings=settings, graph=graph, bus=bus)


def on(p):
    app.state.pipeline = p
    return p


def signed(payload: dict, secret: str = SECRET) -> dict:
    raw = json.dumps(payload).encode()
    sig = "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    return {"content": raw, "headers": {"X-Signature": sig, "Content-Type": "application/json"}}


def next_event(ws) -> dict:
    while True:
        msg = ws.receive_json()
        if msg["type"] == "event":
            return msg["event"]


ALICE, BOB = {"X-User-Id": "alice"}, {"X-User-Id": "bob"}


def test_agent_reply_reaches_websocket_on_other_pod(pipeline, redis_server, client):
    pod_a, pod_b = pod(pipeline, redis_server), pod(pipeline, redis_server)

    on(pod_a)
    r = client.post("/chat", json={"question": "Can I talk to a real person?", "conversation_id": "conv-9"}, headers=ALICE)
    assert r.json()["status"] == "handed_off"
    ticket = client.post("/conversations/conv-9/ws-ticket", headers=ALICE).json()["ticket"]

    with client.websocket_connect(f"/conversations/conv-9/ws?ticket={ticket}") as ws:  # held by pod A
        assert next_event(ws)["type"] == "status"

        on(pod_b)  # the desk's webhook lands on pod B
        assert client.post("/agent/webhook", **signed({"conversation_id": "conv-9", "type": "agent_joined", "agent_name": "Ana"})).status_code == 202
        assert client.post("/agent/webhook", **signed({"conversation_id": "conv-9", "type": "agent_message", "text": "What's wrong?", "agent_name": "Ana"})).status_code == 202
        assert next_event(ws)["type"] == "agent_joined"
        msg = next_event(ws)
        assert msg["type"] == "agent_message" and msg["data"]["text"] == "What's wrong?"

        # the user answers over the same socket -> desk (redacted) + echoed to the user's tabs
        ws.send_json({"type": "user_message", "text": "VPN fails, card 4111 1111 1111 1111"})
        echo = next_event(ws)
        assert echo["type"] == "user_message" and "[REDACTED:card]" in echo["data"]["text"]
    assert pipeline.live_agent.user_messages[-1] == ("conv-9", "VPN fails, card [REDACTED:card]")


def test_reconnect_resumes_with_last_event_id(pipeline, redis_server, client):
    p = on(pod(pipeline, redis_server))
    client.post("/handoff", json={"question": "help", "conversation_id": "conv-r"}, headers=ALICE)
    ticket = client.post("/conversations/conv-r/ws-ticket", headers=ALICE).json()["ticket"]
    with client.websocket_connect(f"/conversations/conv-r/ws?ticket={ticket}") as ws:
        last = next_event(ws)["id"]  # status
    # offline for a moment: the agent writes twice
    for text in ("first", "second"):
        client.post("/agent/webhook", **signed({"conversation_id": "conv-r", "type": "agent_message", "text": text}))
    ticket = client.post("/conversations/conv-r/ws-ticket", headers=ALICE).json()["ticket"]
    with client.websocket_connect(f"/conversations/conv-r/ws?ticket={ticket}&last_event_id={last}") as ws:
        assert [next_event(ws)["data"]["text"] for _ in range(2)] == ["first", "second"]


def test_other_user_cannot_take_over(pipeline, redis_server, client):
    """The old bug: ownership was only checked in agent mode and was claimable later.
    Now every entry point claims or rejects, in bot mode too."""
    on(pod(pipeline, redis_server))
    assert client.post("/chat", json={"question": "VPN client?", "conversation_id": "conv-a"}, headers=ALICE).status_code == 200
    # bot mode: bob can't post into alice's conversation, nor hand it off, nor subscribe
    assert client.post("/chat", json={"question": "hi", "conversation_id": "conv-a"}, headers=BOB).status_code == 404
    assert client.post("/chat/stream", json={"question": "hi", "conversation_id": "conv-a"}, headers=BOB).status_code == 404
    assert client.post("/handoff", json={"question": "hi", "conversation_id": "conv-a"}, headers=BOB).status_code == 404
    assert client.post("/conversations/conv-a/ws-ticket", headers=BOB).status_code == 404
    # after a handoff, still no way in
    client.post("/handoff", json={"question": "human please", "conversation_id": "conv-a"}, headers=ALICE)
    assert client.post("/chat", json={"question": "hi", "conversation_id": "conv-a"}, headers=BOB).status_code == 404


def test_websocket_rejects_bad_or_reused_ticket(pipeline, redis_server, client):
    on(pod(pipeline, redis_server))
    client.post("/handoff", json={"question": "help", "conversation_id": "conv-t"}, headers=ALICE)
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/conversations/conv-t/ws?ticket=forged"):
            pass
    assert exc.value.code == 4401
    ticket = client.post("/conversations/conv-t/ws-ticket", headers=ALICE).json()["ticket"]
    with client.websocket_connect(f"/conversations/conv-t/ws?ticket={ticket}") as ws:
        next_event(ws)
    with pytest.raises(WebSocketDisconnect):  # tickets are single use
        with client.websocket_connect(f"/conversations/conv-t/ws?ticket={ticket}"):
            pass


def test_require_user_id(pipeline, redis_server, client):
    on(pod(pipeline, redis_server, require_user_id=True))
    assert client.post("/chat", json={"question": "VPN client?", "conversation_id": "c"}).status_code == 401
    assert client.post("/chat", json={"question": "VPN client?"}).status_code == 200  # no conversation: fine


def test_webhook_rules(pipeline, redis_server, client):
    on(pod(pipeline, redis_server))
    payload = {"conversation_id": "conv-x", "type": "agent_message", "text": "hi"}
    assert client.post("/agent/webhook", **signed(payload, secret="wrong")).status_code == 401
    assert client.post("/agent/webhook", content=json.dumps(payload)).status_code == 401
    assert client.post("/agent/webhook", **signed(payload)).status_code == 409  # never handed off
    left = {"conversation_id": "conv-x", "type": "agent_left"}
    assert client.post("/agent/webhook", **signed(left)).status_code == 409  # creates no stream for unknown ids


def test_agent_left_returns_conversation_to_bot(pipeline, redis_server, client):
    on(pod(pipeline, redis_server))
    client.post("/handoff", json={"question": "help", "conversation_id": "conv-l"}, headers=ALICE)
    r = client.post("/chat", json={"question": "still there?", "conversation_id": "conv-l"}, headers=ALICE)
    assert r.json()["status"] == "forwarded_to_agent"
    client.post("/agent/webhook", **signed({"conversation_id": "conv-l", "type": "agent_left"}))
    r = client.post("/chat", json={"question": "How often do engineers get a new laptop?", "conversation_id": "conv-l"}, headers=ALICE)
    assert r.json()["status"] == "answered"


def test_webhook_disabled_without_secret(pipeline, client):
    on(pipeline)
    assert client.post("/agent/webhook", **signed({"conversation_id": "c", "type": "agent_left"})).status_code == 503
