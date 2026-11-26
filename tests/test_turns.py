"""Save-first turns: the question is stored as 'processing' before the pipeline runs, a duplicate
message_id is blocked by the UNIQUE constraint, and the sweep job frees rows of dead workers.

Store tests run against the in-memory store and (with RAG_TEST_DATABASE_URL) real PostgreSQL.
"""

import asyncio
import dataclasses

import pytest

from northwind_rag import api as api_module
from northwind_rag.api import app
from northwind_rag.history import InMemoryConversationStore, PostgresConversationStore
from tests.test_history import ALICE, api, needs_pg, pg_url, reset_database, store  # noqa: F401  (fixtures)

W1, W2 = "pod-a:1:aaa", "pod-b:1:bbb"


async def make_stale(store, message_id: int, seconds: int = 600) -> None:
    """Pretend the worker answering `message_id` stopped heart-beating `seconds` ago."""
    if isinstance(store, InMemoryConversationStore):
        from datetime import timedelta

        store._turns[message_id]["heartbeat_at"] -= timedelta(seconds=seconds)
    else:
        await store._fetch("UPDATE messages SET heartbeat_at = now() - make_interval(secs => %s) WHERE id = %s",
                           (seconds, message_id))


async def state_of(store, conversation_id: str) -> list[tuple[str, str | None]]:
    return [(m.role, m.state) for m in await store.messages(conversation_id)]


# ============================================================================ store contract
async def test_question_is_saved_first_as_processing(store):
    await store.claim("t1", "alice")
    claim = await store.begin_turn("t1", "How do I reset my password?", owner=W1,
                                   metadata={"pii_redacted": []}, idempotency_key="user:m1")
    assert claim.outcome == "claimed" and claim.attempt == 1
    assert await state_of(store, "t1") == [("user", "processing")]
    page, _ = await store.conversations("alice")
    assert page[0].title == "How do I reset my password?"


async def test_duplicate_is_processing_then_done(store):
    await store.claim("t2", "alice")
    first = await store.begin_turn("t2", "q", owner=W1, idempotency_key="user:m1")
    assert (await store.begin_turn("t2", "q", owner=W2, idempotency_key="user:m1")).outcome == "processing"
    assert await store.finish_turn(first.message_id, W1, first.attempt)
    assert (await store.begin_turn("t2", "q", owner=W2, idempotency_key="user:m1")).outcome == "done"
    assert len(await store.messages("t2")) == 1  # never stored twice


async def test_concurrent_duplicates_have_exactly_one_winner(store):
    await store.claim("t3", "alice")
    claims = await asyncio.gather(*(store.begin_turn("t3", "q", owner=f"w{i}", idempotency_key="user:m1")
                                    for i in range(12)))
    assert sorted(c.outcome for c in claims) == ["claimed"] + ["processing"] * 11


async def test_failed_question_is_reclaimed_and_old_attempt_is_fenced(store):
    await store.claim("t4", "alice")
    a1 = await store.begin_turn("t4", "q", owner=W1, idempotency_key="user:m1")
    assert await store.finish_turn(a1.message_id, W1, a1.attempt, failed=True, error="boom")
    a2 = await store.begin_turn("t4", "q", owner=W2, idempotency_key="user:m1")
    assert a2.outcome == "claimed" and a2.message_id == a1.message_id and a2.attempt == 2
    # the first attempt can't touch the row any more, even from the same owner id
    assert not await store.heartbeat(a1.message_id, W1, a1.attempt)
    assert not await store.finish_turn(a1.message_id, W2, a1.attempt)
    assert await store.heartbeat(a2.message_id, W2, a2.attempt)


async def test_gives_up_after_max_attempts(store):
    await store.claim("t5", "alice")
    for attempt in (1, 2):
        c = await store.begin_turn("t5", "q", owner=W1, idempotency_key="user:m1", max_attempts=2)
        assert c.outcome == "claimed" and c.attempt == attempt
        await store.finish_turn(c.message_id, W1, c.attempt, failed=True)
    assert (await store.begin_turn("t5", "q", owner=W1, idempotency_key="user:m1", max_attempts=2)).outcome == "gave_up"


async def test_sweep_fails_only_stale_rows(store):
    await store.claim("t6", "alice")
    dead = await store.begin_turn("t6", "dead pod", owner=W1, idempotency_key="user:a")
    live = await store.begin_turn("t6", "live pod", owner=W2, idempotency_key="user:b")
    await make_stale(store, dead.message_id)
    assert await store.sweep_stuck(90) == 1
    assert await state_of(store, "t6") == [("user", "failed"), ("user", "processing")]
    assert not await store.heartbeat(dead.message_id, W1, dead.attempt)  # a zombie learns it lost the row
    assert await store.heartbeat(live.message_id, W2, live.attempt)
    assert (await store.begin_turn("t6", "dead pod", owner=W2, idempotency_key="user:a")).outcome == "claimed"
    assert await store.sweep_stuck(90) == 0


async def test_release_owner_only_touches_own_rows(store):
    await store.claim("t7", "alice")
    await store.begin_turn("t7", "mine", owner=W1, idempotency_key="user:a")
    await store.begin_turn("t7", "theirs", owner=W2, idempotency_key="user:b")
    assert await store.release_owner(W1) == 1
    assert await state_of(store, "t7") == [("user", "failed"), ("user", "processing")]


async def test_without_message_id_every_question_is_new(store):
    await store.claim("t8", "alice")
    a = await store.begin_turn("t8", "q", owner=W1)
    b = await store.begin_turn("t8", "q", owner=W1)
    assert a.outcome == b.outcome == "claimed" and a.message_id != b.message_id


async def test_untracked_row_counts_as_done(store):
    """A message forwarded to a live agent (plain append) is not generated again by the bot."""
    await store.claim("t9", "alice")
    await store.append("t9", "user", "to the agent", status="forwarded_to_agent", idempotency_key="user:m1")
    assert (await store.begin_turn("t9", "to the agent", owner=W1, idempotency_key="user:m1")).outcome == "done"


@needs_pg
def test_migration_marks_pre_0002_aborted_questions_failed(pg_url, monkeypatch):
    import psycopg

    from northwind_rag import migrate as m

    reset_database(pg_url)
    everything = m.migration_files()
    monkeypatch.setattr(m, "migration_files", lambda: everything[:1])  # a database still on 0001
    m.migrate(pg_url)
    with psycopg.connect(pg_url, autocommit=True) as c:
        c.execute("INSERT INTO conversations (id, user_id) VALUES ('old', 'alice')")
        c.execute("""INSERT INTO messages (conversation_id, role, content, metadata, idempotency_key) VALUES
                     ('old', 'user', 'left mid-answer', '{"aborted": true}', 'user:m1'),
                     ('old', 'user', 'answered', '{}', 'user:m2')""")
    monkeypatch.setattr(m, "migration_files", lambda: everything)
    assert m.migrate(pg_url) == ["0002_message_processing.sql"]
    with psycopg.connect(pg_url) as c:
        rows = c.execute("SELECT idempotency_key, state FROM messages ORDER BY id").fetchall()
    assert rows == [("user:m1", "failed"), ("user:m2", None)]  # the aborted one can be retried


# ======================================================================================= API
def with_settings(p, **update):
    p2 = dataclasses.replace(p, settings=p.settings.model_copy(update=update))
    app.state.pipeline = p2
    return p2


async def test_question_is_in_the_database_before_the_pipeline_runs(api, monkeypatch):
    p, client = api
    real_ask = type(p).ask
    seen = {}

    async def ask(self, question, **kw):
        seen["rows"] = await state_of(self.history, "a1")
        return await real_ask(self, question, **kw)

    monkeypatch.setattr(type(p), "ask", ask)
    r = await client.post("/chat", headers=ALICE, json={
        "question": "How often do engineers get a new laptop?", "conversation_id": "a1", "message_id": "m-1"})
    assert r.status_code == 200 and r.json()["status"] == "answered"
    assert seen["rows"] == [("user", "processing")]
    assert await state_of(p.history, "a1") == [("user", "done"), ("assistant", None)]


async def test_duplicate_while_generating_is_409_then_replayed(api, monkeypatch):
    p, client = api
    real_ask = type(p).ask
    entered, release, calls = asyncio.Event(), asyncio.Event(), []

    async def slow_ask(self, question, **kw):
        calls.append(question)
        entered.set()
        await release.wait()
        return await real_ask(self, question, **kw)

    monkeypatch.setattr(type(p), "ask", slow_ask)
    body = {"question": "How long is my account locked?", "conversation_id": "a2", "message_id": "m-1"}
    first = asyncio.create_task(client.post("/chat", headers=ALICE, json=body))
    await entered.wait()
    dup = await client.post("/chat", headers=ALICE, json=body)  # double click while generating
    assert dup.status_code == 409 and dup.json()["error"] == "in_progress" and dup.headers["Retry-After"]
    dup_stream = await client.post("/chat/stream", headers=ALICE, json=body)
    assert dup_stream.status_code == 409
    release.set()
    answer = (await first).json()
    again = (await client.post("/chat", headers=ALICE, json=body)).json()
    assert again["replayed"] and again["answer"] == answer["answer"]
    assert len(calls) == 1  # one generation paid for


async def test_question_of_a_dead_pod_is_swept_and_retried(api):
    p, client = api
    body = {"question": "How long is my account locked?", "conversation_id": "a3", "message_id": "m-1"}
    await p.history.claim("a3", "alice")
    # a pod saved the question and died mid-answer
    crashed = await p.history.begin_turn("a3", body["question"], owner="dead-pod:1:x", idempotency_key="user:m-1")
    assert (await client.post("/chat", headers=ALICE, json=body)).status_code == 409  # still looks alive
    await make_stale(p.history, crashed.message_id)
    assert await p.history.sweep_stuck(p.settings.turn_stale_after_s) == 1
    r = await client.post("/chat", headers=ALICE, json=body)  # client retry
    assert r.status_code == 200 and "15 minutes" in r.json()["answer"]
    assert await state_of(p.history, "a3") == [("user", "done"), ("assistant", None)]


async def test_heartbeat_keeps_a_long_answer_from_being_swept(api, monkeypatch):
    p, client = api
    p = with_settings(p, turn_heartbeat_s=0.05, turn_stale_after_s=0.3)
    real_ask = type(p).ask

    async def slow_ask(self, question, **kw):
        await asyncio.sleep(0.8)  # far longer than the stale window
        return await real_ask(self, question, **kw)

    async def sweeper():
        swept = 0
        for _ in range(8):
            await asyncio.sleep(0.1)
            swept += await p.history.sweep_stuck(0.3)
        return swept

    monkeypatch.setattr(type(p), "ask", slow_ask)
    body = {"question": "How long is my account locked?", "conversation_id": "a4", "message_id": "m-1"}
    r, swept = await asyncio.gather(client.post("/chat", headers=ALICE, json=body), sweeper())
    assert swept == 0 and r.status_code == 200
    assert await state_of(p.history, "a4") == [("user", "done"), ("assistant", None)]


async def test_worker_that_lost_its_row_does_not_write_an_answer(api, monkeypatch):
    p, client = api
    p = with_settings(p, turn_heartbeat_s=30)  # heartbeat never fires during this test
    real_ask = type(p).ask

    async def ask_then_get_swept(self, question, **kw):
        result = await real_ask(self, question, **kw)
        rows = await self.history.messages("a5")
        await make_stale(self.history, rows[0].id)
        await self.history.sweep_stuck(1)  # e.g. a long GC pause / network partition
        return result

    monkeypatch.setattr(type(p), "ask", ask_then_get_swept)
    body = {"question": "How long is my account locked?", "conversation_id": "a5", "message_id": "m-1"}
    await client.post("/chat", headers=ALICE, json=body)
    assert await state_of(p.history, "a5") == [("user", "failed")]  # no orphan answer
    monkeypatch.setattr(type(p), "ask", real_ask)
    assert (await client.post("/chat", headers=ALICE, json=body)).status_code == 200  # retry works
    assert await state_of(p.history, "a5") == [("user", "done"), ("assistant", None)]


async def test_broken_stream_marks_question_failed_and_retry_answers(api, monkeypatch):
    p, client = api
    real_stream = type(p).stream

    async def broken_stream(self, question, **kw):
        yield {"event": "delta", "data": {"text": "Your acc"}}
        raise RuntimeError("LLM connection reset")

    monkeypatch.setattr(type(p), "stream", broken_stream)
    body = {"question": "How long is my account locked?", "conversation_id": "a6", "message_id": "m-1"}
    with pytest.raises(RuntimeError):
        await client.post("/chat/stream", headers=ALICE, json=body)
    assert await state_of(p.history, "a6") == [("user", "failed")]
    monkeypatch.setattr(type(p), "stream", real_stream)
    r = await client.post("/chat/stream", headers=ALICE, json=body)
    assert "event: done" in r.text
    msgs = await p.history.messages("a6")
    assert [(m.role, m.state) for m in msgs] == [("user", "done"), ("assistant", None)]
    assert "15 minutes" in msgs[1].content


async def test_pipeline_error_frees_the_question(api, monkeypatch):
    p, client = api

    async def failing_ask(self, question, **kw):
        raise TimeoutError

    monkeypatch.setattr(type(p), "ask", failing_ask)
    body = {"question": "How long is my account locked?", "conversation_id": "a7", "message_id": "m-1"}
    assert (await client.post("/chat", headers=ALICE, json=body)).status_code == 503
    assert await state_of(p.history, "a7") == [("user", "failed")]


async def test_shutdown_releases_this_workers_questions(api):
    p, _ = api
    await p.history.claim("a8", "alice")
    await p.history.begin_turn("a8", "mine", owner=api_module.WORKER_ID, idempotency_key="user:a")
    await p.history.begin_turn("a8", "other worker", owner="other:2:y", idempotency_key="user:b")
    assert await p.history.release_owner(api_module.WORKER_ID) == 1
    assert await state_of(p.history, "a8") == [("user", "failed"), ("user", "processing")]


async def test_history_api_shows_state(api):
    p, client = api
    await client.post("/chat", headers=ALICE, json={"question": "VPN client?", "conversation_id": "a9", "message_id": "m"})
    msgs = (await client.get("/conversations/a9/messages", headers=ALICE)).json()["messages"]
    assert [m["state"] for m in msgs] == ["done", None]
