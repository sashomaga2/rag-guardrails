"""Concurrency limits: protect the LLM providers, the budget and other users from overload.

Two implementations behind one interface:

LocalLimiter  asyncio.Semaphore per key. Correct only inside ONE process: with 3 pods and a
              limit of 20, you really get 60. Fine for local runs and tests.
RedisLimiter  a distributed semaphore shared by all pods. Each holder is a member of a Redis
              sorted set scored by its lease expiry. Acquire = one atomic Lua script (drop
              expired leases, check count, add). A heartbeat renews the lease while the slot
              is held, so a pod that crashes mid-request frees its slots after `lease_ttl`.

Both raise CapacityExceeded when no slot frees up within `wait_s`: callers answer
503 + Retry-After (backpressure) instead of queueing without bound.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import uuid
from typing import AsyncIterator, Protocol

log = logging.getLogger(__name__)


class CapacityExceeded(Exception):
    def __init__(self, key: str, limit: int, retry_after_s: int = 2):
        super().__init__(f"no free slot for '{key}' (limit {limit})")
        self.key = key
        self.limit = limit
        self.retry_after_s = retry_after_s


class Limiter(Protocol):
    def slot(self, key: str, limit: int, wait_s: float) -> contextlib.AbstractAsyncContextManager[None]: ...

    async def ping(self) -> bool: ...

    async def in_use(self, key: str) -> int: ...


# ---------------------------------------------------------------------------------- local
class LocalLimiter:
    def __init__(self) -> None:
        self._sems: dict[str, tuple[int, asyncio.Semaphore]] = {}
        self._held: dict[str, int] = {}

    def _sem(self, key: str, limit: int) -> asyncio.Semaphore:
        existing = self._sems.get(key)
        if existing is None or existing[0] != limit:
            if existing is not None and self._held.get(key):
                return existing[1]  # don't swap a semaphore that has holders
            existing = (limit, asyncio.Semaphore(limit))
            self._sems[key] = existing
        return existing[1]

    @contextlib.asynccontextmanager
    async def slot(self, key: str, limit: int, wait_s: float) -> AsyncIterator[None]:
        sem = self._sem(key, limit)
        if wait_s <= 0:  # try-acquire: wait_for(timeout=0) would fail even when a slot is free
            if sem.locked():
                raise CapacityExceeded(key, limit)
            await sem.acquire()
        else:
            try:
                await asyncio.wait_for(sem.acquire(), timeout=wait_s)
            except TimeoutError:
                raise CapacityExceeded(key, limit) from None
        self._held[key] = self._held.get(key, 0) + 1
        try:
            yield
        finally:
            self._held[key] -= 1
            sem.release()

    async def ping(self) -> bool:
        return True

    async def in_use(self, key: str) -> int:
        return self._held.get(key, 0)


# ---------------------------------------------------------------------------------- redis
_ACQUIRE = """
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
if redis.call('ZCARD', KEYS[1]) < tonumber(ARGV[2]) then
  redis.call('ZADD', KEYS[1], now + tonumber(ARGV[1]), ARGV[3])
  redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[1]) * 2)
  return 1
end
return 0
"""

_RENEW = """
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local updated = redis.call('ZADD', KEYS[1], 'XX', 'CH', now + tonumber(ARGV[1]), ARGV[2])
redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[1]) * 2)
return updated
"""

_COUNT = """
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
return redis.call('ZCARD', KEYS[1])
"""


class RedisLimiter:
    """Distributed semaphore. Uses Redis server time, so pod clock skew does not matter."""

    def __init__(self, redis, prefix: str = "northwind:limits:", lease_ttl_s: float = 30.0, poll_s: float = 0.05):
        self.redis = redis
        self.prefix = prefix
        self.lease_ms = int(lease_ttl_s * 1000)
        self.poll_s = poll_s
        self._acquire = redis.register_script(_ACQUIRE)
        self._renew = redis.register_script(_RENEW)
        self._count = redis.register_script(_COUNT)

    @classmethod
    def from_url(cls, url: str, **kw) -> "RedisLimiter":
        from redis.asyncio import Redis

        return cls(Redis.from_url(url, socket_timeout=2, socket_connect_timeout=2), **kw)

    async def _heartbeat(self, key: str, token: str) -> None:
        while True:
            await asyncio.sleep(self.lease_ms / 3000)
            try:
                if not await self._renew(keys=[key], args=[self.lease_ms, token]):
                    log.error("lease for %s lost (expired before renewal)", key)
                    return
            except Exception:  # transient Redis error: keep trying until the lease runs out
                log.warning("lease renewal failed for %s", key, exc_info=True)

    @contextlib.asynccontextmanager
    async def slot(self, key: str, limit: int, wait_s: float) -> AsyncIterator[None]:
        rkey, token = self.prefix + key, uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait_s
        while not await self._acquire(keys=[rkey], args=[self.lease_ms, limit, token]):
            if loop.time() >= deadline:
                raise CapacityExceeded(key, limit)
            await asyncio.sleep(self.poll_s * (0.5 + random.random()))  # jitter avoids thundering herd
        heartbeat = asyncio.create_task(self._heartbeat(rkey, token))
        try:
            yield
        finally:
            heartbeat.cancel()
            try:
                await self.redis.zrem(rkey, token)
            except Exception:  # lease expiry frees it anyway
                log.warning("could not release %s; it frees itself after the lease TTL", key, exc_info=True)

    async def ping(self) -> bool:
        try:
            return bool(await self.redis.ping())
        except Exception:
            return False

    async def in_use(self, key: str) -> int:
        return int(await self._count(keys=[self.prefix + key], args=[]))


def build_limiter(redis_url: str | None, lease_ttl_s: float) -> Limiter:
    if redis_url:
        return RedisLimiter.from_url(redis_url, lease_ttl_s=lease_ttl_s)
    return LocalLimiter()
