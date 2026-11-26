"""Conversation bus: delivers live-agent messages to whichever pod holds the user's WebSocket.

Delivery
--------
With N API replicas the user's WebSocket lives on pod A, but the desk's webhook can land on
pod B. Nothing routes to a pod: every conversation has a Redis Stream
(`northwind:conv:<id>:events`). Pod B XADDs; pod A reads and pushes.

Shared reader per pod
---------------------
A pod does NOT open one blocking XREAD connection per user. It runs ONE reader task that
XREADs all conversations its users are connected to, in a single multi-key `XREAD BLOCK`, and
fans each entry out to the local subscribers' queues. 5,000 open sockets on a pod = 1 Redis
connection, not 5,000. When a new conversation is subscribed, the reader is woken through a
per-pod control stream that is part of the same XREAD, so it picks up the new key at once.

A new subscriber first replays its backlog (XRANGE after its Last-Event-ID), then attaches
to the shared reader. Ids are compared numerically, so overlap between replay and live
delivery is dropped: nothing lost, nothing duplicated. A subscriber whose queue overflows (slow
client) is cut off; it reconnects and resumes from its last id.

Why Streams and not Pub/Sub: Pub/Sub drops messages published while a pod restarts or a
client reconnects. Stream entries have ids, so resuming is exact. Streams are capped (MAXLEN)
and expire with the conversation (TTL).

Ownership
---------
`claim()` is the single gate for every conversation operation. It is one atomic Lua script:
  - no owner and no history  -> the caller becomes the owner
  - owner == caller          -> allowed; TTLs of owner, mode and stream are refreshed together
  - owner is someone else    -> denied
  - no owner BUT history     -> denied (orphaned: the owner key expired but the messages still
                                exist; never hand someone else's history to a new user), unless
                                the caller passes verified=True because PostgreSQL, the durable
                                owner record, already confirmed them
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal, Protocol

log = logging.getLogger(__name__)

Mode = Literal["bot", "agent"]
EventType = Literal["status", "agent_joined", "agent_message", "agent_left", "user_message"]


@dataclass(frozen=True)
class ConversationEvent:
    id: str  # "<ms>-<seq>", increasing; used for resume (last_event_id)
    type: EventType
    data: dict

    def to_dict(self) -> dict:
        return {"id": self.id, "type": self.type, "data": self.data}


class SubscriberTooSlow(Exception):
    """The client did not keep up; it should reconnect with its last event id."""


def _id_key(event_id: str | None) -> tuple[int, int]:
    if not event_id:
        return (0, 0)
    ms, _, seq = event_id.partition("-")
    try:
        return (int(ms), int(seq or 0))
    except ValueError:
        return (0, 0)


class ConversationBus(Protocol):
    async def claim(self, conversation_id: str, user_id: str, verified: bool = False) -> bool: ...
    async def forget(self, conversation_id: str) -> None: ...
    async def publish(self, conversation_id: str, type: EventType, data: dict) -> str: ...
    def subscribe(self, conversation_id: str, last_id: str | None = None,
                  keepalive_s: float = 15.0) -> AsyncIterator[ConversationEvent | None]: ...
    async def get_mode(self, conversation_id: str) -> Mode: ...
    async def set_mode(self, conversation_id: str, mode: Mode) -> None: ...
    async def issue_ticket(self, conversation_id: str, user_id: str, ttl_s: int = 30) -> str: ...
    async def redeem_ticket(self, ticket: str, conversation_id: str) -> str | None: ...
    async def close(self) -> None: ...


# ---------------------------------------------------------------------- local fan-out core
@dataclass
class _Subscriber:
    queue: asyncio.Queue
    cursor: tuple[int, int]  # highest id delivered to this subscriber
    overflowed: bool = False


class _FanOut:
    """Local subscribers of one pod, keyed by conversation. Shared by both bus implementations."""

    def __init__(self, queue_size: int) -> None:
        self.queue_size = queue_size
        self.subs: dict[str, list[_Subscriber]] = defaultdict(list)

    def add(self, conversation_id: str, cursor: tuple[int, int]) -> _Subscriber:
        sub = _Subscriber(asyncio.Queue(self.queue_size), cursor)
        self.subs[conversation_id].append(sub)
        return sub

    def remove(self, conversation_id: str, sub: _Subscriber) -> None:
        lst = self.subs.get(conversation_id, [])
        if sub in lst:
            lst.remove(sub)
        if not lst:
            self.subs.pop(conversation_id, None)

    def dispatch(self, conversation_id: str, ev: ConversationEvent) -> None:
        key = _id_key(ev.id)
        for sub in list(self.subs.get(conversation_id, [])):
            if key <= sub.cursor or sub.overflowed:
                continue  # already delivered by its replay
            try:
                sub.queue.put_nowait(ev)
                sub.cursor = key
            except asyncio.QueueFull:
                # drop the backlog and wake the consumer with a sentinel: it disconnects and the
                # client resumes from the last id it actually received
                sub.overflowed = True
                while not sub.queue.empty():
                    sub.queue.get_nowait()
                sub.queue.put_nowait(None)


async def _drain(sub: _Subscriber, keepalive_s: float) -> AsyncIterator[ConversationEvent | None]:
    while True:
        try:
            ev = await asyncio.wait_for(sub.queue.get(), timeout=keepalive_s)
        except TimeoutError:
            yield None  # idle: caller sends a ping
            continue
        if sub.overflowed:
            raise SubscriberTooSlow()
        yield ev


# ------------------------------------------------------------------------------- local bus
class LocalBus:
    """Single-process bus (local runs, tests). Same semantics as RedisStreamBus."""

    def __init__(self, max_events: int = 500, queue_size: int = 100) -> None:
        self.max_events = max_events
        self._events: dict[str, list[ConversationEvent]] = defaultdict(list)
        self._mode: dict[str, Mode] = {}
        self._owner: dict[str, str] = {}
        self._tickets: dict[str, tuple[str, str, float]] = {}
        self._fan = _FanOut(queue_size)
        self._last_ms = 0
        self._seq = 0

    def _next_id(self) -> str:
        ms = int(time.time() * 1000)
        if ms <= self._last_ms:
            ms, self._seq = self._last_ms, self._seq + 1
        else:
            self._seq = 0
        self._last_ms = ms
        return f"{ms}-{self._seq}"

    async def claim(self, conversation_id: str, user_id: str, verified: bool = False) -> bool:
        owner = self._owner.get(conversation_id)
        if owner is None:
            if self._events.get(conversation_id) and not verified:
                return False  # orphaned history (unless the durable store confirmed this owner)
            self._owner[conversation_id] = user_id
            return True
        return owner == user_id

    async def publish(self, conversation_id: str, type: EventType, data: dict) -> str:
        ev = ConversationEvent(self._next_id(), type, data)
        events = self._events[conversation_id]
        events.append(ev)
        del events[: -self.max_events]
        self._fan.dispatch(conversation_id, ev)
        return ev.id

    async def subscribe(self, conversation_id: str, last_id: str | None = None,
                        keepalive_s: float = 15.0) -> AsyncIterator[ConversationEvent | None]:
        cursor = _id_key(last_id)
        backlog = [e for e in self._events.get(conversation_id, []) if _id_key(e.id) > cursor]
        sub = self._fan.add(conversation_id, _id_key(backlog[-1].id) if backlog else cursor)
        try:
            for ev in backlog:
                yield ev
            async for ev in _drain(sub, keepalive_s):
                yield ev
        finally:
            self._fan.remove(conversation_id, sub)

    async def forget(self, conversation_id: str) -> None:
        for d in (self._events, self._mode, self._owner):
            d.pop(conversation_id, None)

    async def get_mode(self, conversation_id: str) -> Mode:
        return self._mode.get(conversation_id, "bot")

    async def set_mode(self, conversation_id: str, mode: Mode) -> None:
        self._mode[conversation_id] = mode

    async def issue_ticket(self, conversation_id: str, user_id: str, ttl_s: int = 30) -> str:
        ticket = secrets.token_urlsafe(24)
        self._tickets[ticket] = (conversation_id, user_id, time.monotonic() + ttl_s)
        return ticket

    async def redeem_ticket(self, ticket: str, conversation_id: str) -> str | None:
        entry = self._tickets.pop(ticket, None)  # single use
        if not entry or entry[0] != conversation_id or entry[2] < time.monotonic():
            return None
        return entry[1]

    async def close(self) -> None:
        pass


# ------------------------------------------------------------------------------- redis bus
_CLAIM = """
local owner = redis.call('GET', KEYS[1])
if not owner then
  if ARGV[3] ~= '1' and redis.call('EXISTS', KEYS[2]) == 1 then return 0 end
  redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
  return 1
end
if owner ~= ARGV[1] then return 0 end
redis.call('EXPIRE', KEYS[1], ARGV[2])
redis.call('EXPIRE', KEYS[2], ARGV[2])
redis.call('EXPIRE', KEYS[3], ARGV[2])
return 1
"""


class RedisStreamBus:
    def __init__(self, redis, reader_redis=None, prefix: str = "northwind:", ttl_s: int = 24 * 3600,
                 max_events: int = 500, queue_size: int = 100, block_ms: int = 5000, pod_id: str | None = None):
        self.redis = redis  # commands
        self.reader_redis = reader_redis or redis  # the one blocking XREAD connection of this pod
        self.prefix = prefix
        self.ttl_s = ttl_s
        self.max_events = max_events
        self.block_ms = block_ms
        self.pod_id = pod_id or uuid.uuid4().hex[:12]
        self._claim = redis.register_script(_CLAIM)
        self._fan = _FanOut(queue_size)
        self._cursors: dict[str, str] = {}  # stream key -> last id read by the shared reader
        self._reader: asyncio.Task | None = None
        self.reader_iterations = 0  # observability / tests

    @classmethod
    def from_url(cls, url: str, **kw) -> "RedisStreamBus":
        from redis.asyncio import Redis

        commands = Redis.from_url(url, socket_connect_timeout=2, socket_timeout=5)
        reader = Redis.from_url(url, socket_connect_timeout=2, health_check_interval=30)  # no read timeout: BLOCK
        return cls(commands, reader, **kw)

    # -- keys -----------------------------------------------------------------------------
    def _events_key(self, cid: str) -> str:
        return f"{self.prefix}conv:{cid}:events"

    def _owner_key(self, cid: str) -> str:
        return f"{self.prefix}conv:{cid}:owner"

    def _mode_key(self, cid: str) -> str:
        return f"{self.prefix}conv:{cid}:mode"

    @property
    def _wake_key(self) -> str:
        return f"{self.prefix}pod:{self.pod_id}:wake"

    # -- ownership ------------------------------------------------------------------------
    async def claim(self, conversation_id: str, user_id: str, verified: bool = False) -> bool:
        """verified=True: the durable store (Postgres) already confirmed this user as the owner,
        so an orphaned stream may be re-attached to them (never to anyone else)."""
        keys = [self._owner_key(conversation_id), self._events_key(conversation_id), self._mode_key(conversation_id)]
        return bool(await self._claim(keys=keys, args=[user_id, self.ttl_s, "1" if verified else "0"]))

    async def forget(self, conversation_id: str) -> None:
        await self.redis.delete(self._events_key(conversation_id), self._owner_key(conversation_id),
                                self._mode_key(conversation_id))

    # -- publish --------------------------------------------------------------------------
    async def publish(self, conversation_id: str, type: EventType, data: dict) -> str:
        key = self._events_key(conversation_id)
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.xadd(key, {"type": type, "data": json.dumps(data, ensure_ascii=False)},
                      maxlen=self.max_events, approximate=True)
            pipe.expire(key, self.ttl_s)
            pipe.expire(self._owner_key(conversation_id), self.ttl_s)  # history and owner live and die together
            event_id, *_ = await pipe.execute()
        return event_id.decode() if isinstance(event_id, bytes) else event_id

    # -- subscribe (shared reader) ----------------------------------------------------------
    @staticmethod
    def _decode(entry_id, fields) -> ConversationEvent:
        entry_id = entry_id.decode() if isinstance(entry_id, bytes) else entry_id
        f = {(k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
             for k, v in fields.items()}
        return ConversationEvent(entry_id, f["type"], json.loads(f["data"]))

    async def subscribe(self, conversation_id: str, last_id: str | None = None,
                        keepalive_s: float = 15.0) -> AsyncIterator[ConversationEvent | None]:
        key = self._events_key(conversation_id)
        # 1. replay backlog with a normal (non-blocking) command
        start = f"({last_id}" if last_id else "-"
        backlog = [self._decode(i, f) for i, f in await self.redis.xrange(key, min=start, max="+")]
        cursor = _id_key(backlog[-1].id) if backlog else _id_key(last_id)
        # 2. attach to the shared reader, starting where the replay ended
        sub = self._fan.add(conversation_id, cursor)
        if key not in self._cursors:
            self._cursors[key] = backlog[-1].id if backlog else (last_id or "0-0")
        self._ensure_reader()
        await self.redis.xadd(self._wake_key, {"k": key}, maxlen=10, approximate=False)  # wake the reader now
        try:
            for ev in backlog:
                yield ev
            async for ev in _drain(sub, keepalive_s):
                yield ev
        finally:
            self._fan.remove(conversation_id, sub)
            if conversation_id not in self._fan.subs:
                self._cursors.pop(key, None)

    def _ensure_reader(self) -> None:
        if self._reader is None or self._reader.done():
            self._reader = asyncio.create_task(self._read_loop(), name=f"conv-reader-{self.pod_id}")

    async def _read_loop(self) -> None:
        wake_cursor = "$"
        backoff = 0.1
        while self._cursors:
            streams = {self._wake_key: wake_cursor, **self._cursors}
            try:
                result = await self.reader_redis.xread(streams, block=self.block_ms, count=200)
                backoff = 0.1
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning("conversation reader error; retrying", exc_info=True)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
                continue
            self.reader_iterations += 1
            for stream, entries in result or []:
                stream = stream.decode() if isinstance(stream, bytes) else stream
                if stream == self._wake_key:
                    last = entries[-1][0]
                    wake_cursor = last.decode() if isinstance(last, bytes) else last
                    continue
                if stream not in self._cursors:
                    continue  # last subscriber left while we were blocked
                cid = stream[len(self.prefix + "conv:"):-len(":events")]
                for entry_id, fields in entries:
                    ev = self._decode(entry_id, fields)
                    self._cursors[stream] = ev.id
                    self._fan.dispatch(cid, ev)
        self._reader = None  # nobody subscribed: the next subscribe starts a new loop

    # -- state ----------------------------------------------------------------------------
    async def get_mode(self, conversation_id: str) -> Mode:
        mode = await self.redis.get(self._mode_key(conversation_id))
        return (mode.decode() if isinstance(mode, bytes) else mode) or "bot"

    async def set_mode(self, conversation_id: str, mode: Mode) -> None:
        await self.redis.set(self._mode_key(conversation_id), mode, ex=self.ttl_s)

    async def issue_ticket(self, conversation_id: str, user_id: str, ttl_s: int = 30) -> str:
        ticket = secrets.token_urlsafe(24)
        await self.redis.set(f"{self.prefix}ticket:{ticket}", f"{conversation_id}\n{user_id}", ex=ttl_s)
        return ticket

    async def redeem_ticket(self, ticket: str, conversation_id: str) -> str | None:
        raw = await self.redis.getdel(f"{self.prefix}ticket:{ticket}")  # atomic single use
        if not raw:
            return None
        cid, _, user = (raw.decode() if isinstance(raw, bytes) else raw).partition("\n")
        return user if cid == conversation_id else None

    async def close(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
            try:
                await self._reader
            except (asyncio.CancelledError, Exception):
                pass


def build_bus(redis_url: str | None, ttl_s: int = 24 * 3600) -> ConversationBus:
    return RedisStreamBus.from_url(redis_url, ttl_s=ttl_s) if redis_url else LocalBus()
