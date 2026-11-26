"""Conversation history: store contract (in-memory AND real PostgreSQL), migrations, API.

PostgreSQL tests run when RAG_TEST_DATABASE_URL points at an empty database, e.g.
    docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=pg postgres:16-alpine
    RAG_TEST_DATABASE_URL=postgresql://postgres:pg@localhost:5432/postgres pytest tests/test_history.py
Otherwise they are skipped; the in-memory variant always runs.
"""

import asyncio
import dataclasses
import os
import threading

import httpx
import pytest
import pytest_asyncio

from northwind_rag.api import app
from northwind_rag.conversations import LocalBus
from northwind_rag.graph import build_graph
from northwind_rag.history import InMemoryConversationStore, PostgresConversationStore

PG_URL = os.environ.get("RAG_TEST_DATABASE_URL")
needs_pg = pytest.mark.skipif(not PG_URL, reason="RAG_TEST_DATABASE_URL not set")


def reset_database(url: str) -> None:
    import psycopg

    with psycopg.connect(url, autocommit=True) as c:
        c.execute("DROP TABLE IF EXISTS messages, conversations, schema_migrations CASCADE")


@pytest.fixture
def pg_url():
    if not PG_URL:
        pytest.skip("RAG_TEST_DATABASE_URL not set")
    from northwind_rag.migrate import migrate

    reset_database(PG_URL)
    migrate(PG_URL)
    return PG_URL


# loop_scope="function": the connection pool must live and die on the test's own event loop
@pytest_asyncio.fixture(loop_scope="function", params=["memory", pytest.param("postgres", marks=needs_pg)])
async def store(request):
    if request.param == "memory":
        yield InMemoryConversationStore()
        return
    url = request.getfixturevalue("pg_url")
    s = PostgresConversationStore(url, min_size=1, max_size=4)
    yield s
    await s.close()


async def age(store, conversation_id: str, days: int) -> None:
    """Pretend a conversation was last touched `days` ago."""
    if isinstance(store, InMemoryConversationStore):
        from datetime import timedelta

        c = store._convs[conversation_id]
        store._convs[conversation_id] = dataclasses.replace(c, updated_at=c.updated_at - timedelta(days=days))
    else:
        await store._fetch("UPDATE conversations SET updated_at = now() - make_interval(days => %s) WHERE id = %s",
                           (days, conversation_id))


# ============================================================================ store contract
async def test_claim_is_first_come_and_durable(store):
    assert await store.claim("c1", "alice")
    assert await store.claim("c1", "alice")
    assert not await store.claim("c1", "bob")
    assert await store.owner("c1") == "alice" and await store.owner("nope") is None


async def test_append_is_idempotent_and_sets_title(store):
    await store.claim("c2", "alice")
    first = await store.append("c2", "user", "How do I reset my password?", idempotency_key="user:m1")
    again = await store.append("c2", "user", "How do I reset my password?", idempotency_key="user:m1")
    assert first and again is None  # retried request stored once
    await store.append("c2", "assistant", "Use id.northwind.example [passwords#2].", status="answered",
                       metadata={"citations": ["passwords#2"]}, idempotency_key="assistant:m1")
    await store.append("c2", "user", "second question")  # no key: always stored
    msgs = await store.messages("c2")
    assert [m.role for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1].metadata == {"citations": ["passwords#2"]} and msgs[1].status == "answered"
    page, _ = await store.conversations("alice")
    assert page[0].title == "How do I reset my password?"  # first user message, never overwritten
    found = await store.find("c2", "assistant:m1")
    assert found and found.content.startswith("Use id.")


async def test_append_to_unknown_conversation_fails(store):
    with pytest.raises(Exception):
        await store.append("ghost", "user", "hi")


async def test_message_paging_is_chronological(store):
    await store.claim("c3", "alice")
    ids = [await store.append("c3", "user", f"m{i}") for i in range(5)]
    last_two = await store.messages("c3", limit=2)
    assert [m.content for m in last_two] == ["m3", "m4"]
    earlier = await store.messages("c3", before_id=ids[3], limit=2)
    assert [m.content for m in earlier] == ["m1", "m2"]


async def test_conversation_list_keyset_pagination(store):
    for i in range(5):
        await store.claim(f"conv-{i}", "alice")
        await store.append(f"conv-{i}", "user", f"question {i}")
        await asyncio.sleep(0.002)
    await store.claim("bobs", "bob")
    seen, cursor = [], None
    while True:
        page, cursor = await store.conversations("alice", cursor=cursor, limit=2)
        seen += [c.id for c in page]
        if not cursor:
            break
    assert seen == [f"conv-{i}" for i in reversed(range(5))]  # newest first, no gaps, no repeats
    assert "bobs" not in seen


async def test_status_delete_and_purge(store):
    await store.claim("keep", "alice")
    await store.claim("old", "alice")
    await store.append("old", "user", "ancient")
    await store.set_status("keep", "agent", "INC-1")
    page, _ = await store.conversations("alice")
    assert next(c for c in page if c.id == "keep").handoff_ticket == "INC-1"
    await age(store, "old", 120)
    assert await store.purge(90) == 1
    assert await store.owner("old") is None and await store.messages("old") == []
    assert await store.delete("keep") and not await store.delete("keep")


# ================================================================================ migrations
@needs_pg
def test_migrations_idempotent_and_ping_requires_schema(pg_url):
    from northwind_rag.migrate import migrate

    assert migrate(pg_url) == []  # already applied by the fixture
    reset_database(pg_url)

    async def ping():
        s = PostgresConversationStore(pg_url)
        try:
            return await s.ping()
        finally:
            await s.close()

    assert asyncio.run(ping()) is False  # reachable but not migrated -> not ready
    assert migrate(pg_url) == ["0001_conversations.sql", "0002_message_processing.sql"]
    assert asyncio.run(ping()) is True


@needs_pg
def test_concurrent_migrators_apply_once(pg_url):
    from northwind_rag import migrate as m
    from northwind_rag.migrate import migrate

    reset_database(pg_url)
    results, errors = [], []

    def run():
        try:
            results.append(migrate(pg_url))
        except Exception as exc:  # pragma: no cover - would fail the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    n = len(m.migration_files())
    assert not errors and sorted(map(len, results)) == [0, 0, 0, n]  # exactly one runner applied them


@needs_pg
def test_edited_migration_is_rejected(pg_url, monkeypatch):
    from northwind_rag import migrate as m

    real = m.migration_files()
    monkeypatch.setattr(m, "migration_files", lambda: [(v, n, sql + "\n-- edited") for v, n, sql in real])
    with pytest.raises(RuntimeError, match="changed after it was applied"):
        m.migrate(pg_url)


# ======================================================================================= API
@pytest_asyncio.fixture(loop_scope="function", params=["memory", pytest.param("postgres", marks=needs_pg)])
async def api(request, pipeline):
    history = InMemoryConversationStore() if request.param == "memory" else \
        PostgresConversationStore(request.getfixturevalue("pg_url"), max_size=4)
    settings = pipeline.settings.model_copy(update={"agent_webhook_secret": "s"})
    bus = LocalBus()
    graph = build_graph(pipeline.suite, pipeline.kb, settings, pipeline.live_agent, bus)
    p = dataclasses.replace(pipeline, settings=settings, graph=graph, bus=bus, history=history)
    app.state.pipeline = p
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield p, client
    await history.close()


ALICE, BOB = {"X-User-Id": "alice"}, {"X-User-Id": "bob"}


async def test_chat_turn_is_persisted_redacted(api):
    p, client = api
    r = await client.post("/chat", headers=ALICE, json={
        "question": "My card 4111 1111 1111 1111 broke, can I buy a monitor with the home office budget?",
        "conversation_id": "h1", "message_id": "m-1"})
    assert r.json()["status"] == "answered"
    msgs = (await client.get("/conversations/h1/messages", headers=ALICE)).json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert "4111" not in msgs[0]["content"] and "[REDACTED:card]" in msgs[0]["content"]
    assert msgs[1]["status"] == "answered" and msgs[1]["metadata"]["citations"]
    convs = (await client.get("/conversations", headers=ALICE)).json()["conversations"]
    assert convs[0]["id"] == "h1" and "[REDACTED:card]" in convs[0]["title"]


async def test_retry_with_same_message_id_is_replayed_not_regenerated(api, monkeypatch):
    p, client = api
    body = {"question": "How often do engineers get a new laptop?", "conversation_id": "h2", "message_id": "m-9"}
    first = (await client.post("/chat", headers=ALICE, json=body)).json()

    async def must_not_run(*a, **kw):
        raise AssertionError("pipeline ran again for a retried message")

    monkeypatch.setattr(type(p), "ask", must_not_run)
    second = (await client.post("/chat", headers=ALICE, json=body)).json()
    assert second["replayed"] and second["answer"] == first["answer"]
    assert len((await client.get("/conversations/h2/messages", headers=ALICE)).json()["messages"]) == 2


async def test_stream_is_persisted_when_done(api):
    p, client = api
    r = await client.post("/chat/stream", headers=ALICE,
                          json={"question": "How long is my account locked?", "conversation_id": "h3"})
    assert "event: done" in r.text
    msgs = (await client.get("/conversations/h3/messages", headers=ALICE)).json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"] and "15 minutes" in msgs[1]["content"]


async def test_agent_conversation_and_duplicate_webhook(api):
    import hashlib, hmac, json

    p, client = api
    await client.post("/chat", headers=ALICE, json={"question": "I want a human agent", "conversation_id": "h4"})

    def signed(payload):
        raw = json.dumps(payload).encode()
        return {"content": raw, "headers": {"Content-Type": "application/json",
                                            "X-Signature": "sha256=" + hmac.new(b"s", raw, hashlib.sha256).hexdigest()}}

    msg = {"conversation_id": "h4", "type": "agent_message", "text": "Hi, Ana here.", "agent_name": "Ana", "event_id": "desk-77"}
    assert (await client.post("/agent/webhook", **signed(msg))).json().get("event_id")
    assert (await client.post("/agent/webhook", **signed(msg))).json() == {"accepted": True, "duplicate": True}
    pushed = [e async for e in _drain(p.bus.subscribe("h4", keepalive_s=0.05))]
    assert sum(e.type == "agent_message" for e in pushed) == 1  # redelivery not pushed twice

    await client.post("/chat", headers=ALICE, json={"question": "thanks, it works", "conversation_id": "h4"})
    msgs = (await client.get("/conversations/h4/messages", headers=ALICE)).json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "agent", "user"]
    assert msgs[1]["status"] == "handed_off" and msgs[3]["status"] == "forwarded_to_agent"
    convs = (await client.get("/conversations", headers=ALICE)).json()["conversations"]
    assert convs[0]["status"] == "agent" and convs[0]["handoff_ticket"]


async def _drain(sub):
    async for e in sub:
        if e is None:
            return
        yield e


async def test_ownership_survives_redis_loss(api):
    """Redis is a cache now: flush it and the durable owner in Postgres still decides."""
    p, client = api
    await client.post("/chat", headers=ALICE, json={"question": "VPN client?", "conversation_id": "h5"})
    p.bus = LocalBus()  # Redis restarted / evicted: no owner keys any more
    app.state.pipeline = p
    assert (await client.post("/chat", headers=BOB, json={"question": "hi", "conversation_id": "h5"})).status_code == 404
    assert (await client.post("/chat", headers=ALICE, json={"question": "VPN client?", "conversation_id": "h5"})).status_code == 200


async def test_read_and_delete_are_owner_only(api):
    p, client = api
    await client.post("/chat", headers=ALICE, json={"question": "VPN client?", "conversation_id": "h6"})
    assert (await client.get("/conversations/h6/messages", headers=BOB)).status_code == 404
    assert (await client.delete("/conversations/h6", headers=BOB)).status_code == 404
    assert (await client.get("/conversations/never-existed/messages", headers=ALICE)).status_code == 404  # no creation
    assert (await client.delete("/conversations/h6", headers=ALICE)).status_code == 204
    assert (await client.get("/conversations/h6/messages", headers=ALICE)).status_code == 404
    assert await p.history.owner("h6") is None


async def test_bad_cursor_is_400(api):
    _, client = api
    assert (await client.get("/conversations?cursor=garbage", headers=ALICE)).status_code == 400
