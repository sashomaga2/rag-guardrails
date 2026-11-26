import asyncio
import dataclasses

import fakeredis
import httpx
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from northwind_rag.api import app
from northwind_rag.graph import build_graph
from northwind_rag.limits import CapacityExceeded, LocalLimiter, RedisLimiter
from northwind_rag.models import CallPolicy
from northwind_rag.pipeline import Pipeline
from tests.test_streaming_api import parse_sse


async def run_many(limiters, key, limit, n=30, hold_s=0.02, wait_s=5.0):
    """n tasks spread over the given limiters (one per 'pod'); returns peak concurrency."""
    active = peak = 0

    async def worker(i: int):
        nonlocal active, peak
        async with limiters[i % len(limiters)].slot(key, limit, wait_s):
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(hold_s)
            active -= 1

    await asyncio.gather(*(worker(i) for i in range(n)))
    return peak


async def test_local_limiter_caps_concurrency():
    assert await run_many([LocalLimiter()], "k", limit=5) == 5


async def test_local_limiter_rejects_after_wait():
    lim = LocalLimiter()
    async with lim.slot("k", 1, 1):
        with pytest.raises(CapacityExceeded):
            async with lim.slot("k", 1, wait_s=0.05):
                pass
    async with lim.slot("k", 1, 0.05):  # released again
        assert await lim.in_use("k") == 1


@pytest.fixture
def redis_server():
    return fakeredis.FakeServer()


def pod(server, **kw) -> RedisLimiter:
    return RedisLimiter(fakeredis.FakeAsyncRedis(server=server), poll_s=0.005, **kw)


async def test_redis_limit_is_global_across_pods(redis_server):
    pods = [pod(redis_server), pod(redis_server), pod(redis_server)]
    assert await run_many(pods, "llm:generator", limit=4, n=40) == 4  # not 4 per pod
    assert await pods[0].in_use("llm:generator") == 0  # everything released


async def test_redis_rejects_when_full(redis_server):
    a, b = pod(redis_server), pod(redis_server)
    async with a.slot("chats", 1, 1):
        with pytest.raises(CapacityExceeded):
            async with b.slot("chats", 1, wait_s=0.05):
                pass


async def test_crashed_pod_frees_slot_after_lease(redis_server):
    crashed = pod(redis_server, lease_ttl_s=0.2)
    # simulate a pod that died while holding a slot: acquire but never release, no heartbeat
    assert await crashed._acquire(keys=[crashed.prefix + "chats"], args=[crashed.lease_ms, 1, "dead-pod"])
    healthy = pod(redis_server, lease_ttl_s=0.2)
    with pytest.raises(CapacityExceeded):
        async with healthy.slot("chats", 1, wait_s=0.05):
            pass
    async with healthy.slot("chats", 1, wait_s=1.0):  # lease expired -> slot reclaimed
        pass


async def test_heartbeat_keeps_long_holder(redis_server):
    holder, other = pod(redis_server, lease_ttl_s=0.15), pod(redis_server, lease_ttl_s=0.15)
    async with holder.slot("chats", 1, 1):
        await asyncio.sleep(0.5)  # > 3x the lease: only heartbeats keep it
        with pytest.raises(CapacityExceeded):
            async with other.slot("chats", 1, wait_s=0.05):
                pass


# ------------------------------------------------------------------------------- API level
async def post(pipeline, path, json, headers=None):
    app.state.pipeline = pipeline
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        return await c.post(path, json=json, headers=headers or {})


def variant(pipeline, chat=None, policy=None, **settings_overrides) -> Pipeline:
    settings = pipeline.settings.model_copy(update=settings_overrides)
    limiter = policy.limiter if policy else LocalLimiter()
    suite = dataclasses.replace(
        pipeline.suite, chat=chat or pipeline.suite.chat, policy=policy or CallPolicy(limiter=limiter)
    )
    graph = build_graph(suite, pipeline.kb, settings, pipeline.live_agent)
    return Pipeline(settings, suite, pipeline.kb, graph, pipeline.live_agent, limiter)


async def test_overload_returns_503_with_retry_after(pipeline):
    p = variant(pipeline, max_concurrent_chats=1, slot_wait_s=0.05)
    async with p.limiter.slot("chats", 1, 1):  # someone else holds the only slot
        r = await post(p, "/chat", {"question": "How often do engineers get a new laptop?"})
        s = await post(p, "/chat/stream", {"question": "How often do engineers get a new laptop?"})
    assert r.status_code == 503 and r.headers["retry-after"] == "2"
    assert s.status_code == 503  # rejected before the stream starts
    ok = await post(p, "/chat", {"question": "How often do engineers get a new laptop?"})
    assert ok.status_code == 200  # slot freed


async def test_per_user_stream_limit(pipeline):
    p = variant(pipeline, max_streams_per_user=1)
    async with p.limiter.slot("user:alice", 1, 1):
        busy = await post(p, "/chat", {"question": "VPN client?"}, {"X-User-Id": "alice"})
        other = await post(p, "/chat", {"question": "VPN client?"}, {"X-User-Id": "bob"})
    assert busy.status_code == 503 and other.status_code == 200


async def test_slots_released_after_stream(pipeline):
    p = variant(pipeline)
    r = await post(p, "/chat/stream", {"question": "How long is my account locked?"}, {"X-User-Id": "u1"})
    assert r.status_code == 200
    assert await p.limiter.in_use("chats") == 0 and await p.limiter.in_use("user:u1") == 0


class SlowChat(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "slow"

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        raise NotImplementedError

    async def _agenerate(self, messages, stop=None, run_manager=None, **kw):
        await asyncio.sleep(5)
        return ChatResult(generations=[ChatGeneration(message=AIMessage("late"))])


async def test_generation_timeout_fails_closed_in_stream(pipeline):
    policy = CallPolicy(timeouts={"generator": 0.05, "guard": 1.0})
    p = variant(pipeline, chat=SlowChat(), policy=policy)
    r = await post(p, "/chat/stream", {"question": "How often do engineers get a new laptop?"})
    events = parse_sse(r.text)
    names = [e for e, _ in events]
    assert "error" in names and "verified" not in names
    err = next(d for e, d in events if e == "error")
    assert err["code"] == "timeout"
    assert events[-1][1]["status"] == "error"
    assert await p.limiter.in_use("llm:generator") == 0  # slot released on timeout


async def test_generation_timeout_returns_503_on_chat(pipeline):
    p = variant(pipeline, chat=SlowChat(), policy=CallPolicy(timeouts={"generator": 0.05, "guard": 1.0}))
    r = await post(p, "/chat", {"question": "How often do engineers get a new laptop?"})
    assert r.status_code == 503 and r.json()["error"] == "timeout"
