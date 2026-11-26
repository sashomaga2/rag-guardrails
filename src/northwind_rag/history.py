"""Durable conversation history (PostgreSQL) and the durable owner of every conversation.

Who stores what
---------------
PostgreSQL  system of record: conversations, every message (user, assistant, agent, system),
            the owner. Survives restarts, Redis flushes and the 24h stream TTL.
Redis       hot path only: limiter leases, live push streams, a cached copy of the owner.

Rules
-----
- Only redacted text is stored (the same sanitized question the LLM saw).
- Writes are idempotent: a client `message_id` or a desk `event_id` becomes the idempotency
  key, so a retried request or a redelivered webhook is stored once (UNIQUE constraint).
- Ownership: `claim()` inserts the conversation for the caller, or returns whether the caller
  already owns it, in one statement. This is the authoritative check; Redis follows it.
- Retention: `purge(days)` deletes whole conversations not updated for N days (CronJob).
- Save-first turns: `begin_turn()` stores the user's question with state 'processing' BEFORE
  the pipeline runs. The insert is the lock (UNIQUE idempotency key): a duplicate gets
  'processing' (409), 'done' (replay) or re-claims a 'failed' row. While generating, the worker
  renews `heartbeat()`; `sweep_stuck()` (CronJob) fails rows whose worker died, so a retry can
  run. `attempts` is a fencing token: a worker that lost its row can no longer finish it.

PostgresConversationStore  psycopg 3 async + connection pool.
InMemoryConversationStore  same contract, for offline mode and tests.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Protocol

log = logging.getLogger(__name__)

Role = Literal["user", "assistant", "agent", "system"]
ConvStatus = Literal["bot", "agent", "closed"]
TurnState = Literal["processing", "done", "failed"]
TurnOutcome = Literal["claimed", "processing", "done", "gave_up"]


@dataclass(frozen=True)
class StoredMessage:
    id: int
    conversation_id: str
    role: Role
    content: str
    status: str | None
    metadata: dict
    created_at: datetime
    state: TurnState | None = None  # user questions only: processing / done / failed

    def to_dict(self) -> dict:
        return {"id": self.id, "role": self.role, "content": self.content, "status": self.status,
                "state": self.state, "metadata": self.metadata, "created_at": self.created_at.isoformat()}


@dataclass(frozen=True)
class TurnClaim:
    """Result of `begin_turn`.

    claimed     this worker owns the question row: run the pipeline, then finish_turn()
    processing  another request is generating it right now (API: 409)
    done        already answered (API: replay the stored answer)
    gave_up     failed `max_attempts` times; the client should send it as a new message
    """

    outcome: TurnOutcome
    message_id: int | None = None
    attempt: int = 1  # fencing token for heartbeat / finish_turn


@dataclass(frozen=True)
class ConversationInfo:
    id: str
    user_id: str
    status: ConvStatus
    title: str | None
    handoff_ticket: str | None
    created_at: datetime
    updated_at: datetime

    def to_dict(self) -> dict:
        return {"id": self.id, "status": self.status, "title": self.title, "handoff_ticket": self.handoff_ticket,
                "created_at": self.created_at.isoformat(), "updated_at": self.updated_at.isoformat()}


class ConversationStore(Protocol):
    async def claim(self, conversation_id: str, user_id: str) -> bool: ...
    async def owner(self, conversation_id: str) -> str | None: ...
    async def append(self, conversation_id: str, role: Role, content: str, *, status: str | None = None,
                     metadata: dict | None = None, idempotency_key: str | None = None) -> int | None: ...
    async def find(self, conversation_id: str, idempotency_key: str) -> StoredMessage | None: ...
    async def set_status(self, conversation_id: str, status: ConvStatus, handoff_ticket: str | None = None) -> None: ...
    async def messages(self, conversation_id: str, *, before_id: int | None = None, limit: int = 50) -> list[StoredMessage]: ...
    async def conversations(self, user_id: str, *, cursor: str | None = None, limit: int = 20) -> tuple[list[ConversationInfo], str | None]: ...
    async def delete(self, conversation_id: str) -> bool: ...
    async def purge(self, older_than_days: int) -> int: ...
    async def begin_turn(self, conversation_id: str, content: str, *, owner: str, metadata: dict | None = None,
                         idempotency_key: str | None = None, max_attempts: int = 3) -> TurnClaim: ...
    async def heartbeat(self, message_id: int, owner: str, attempt: int) -> bool: ...
    async def finish_turn(self, message_id: int, owner: str, attempt: int, *, failed: bool = False,
                          error: str | None = None) -> bool: ...
    async def release_owner(self, owner: str) -> int: ...
    async def sweep_stuck(self, stale_after_s: float, limit: int = 1000) -> int: ...
    async def ping(self) -> bool: ...
    async def close(self) -> None: ...


TITLE_CHARS = 80
ERROR_CHARS = 500


def _cursor(c: ConversationInfo) -> str:
    return f"{c.updated_at.isoformat()}|{c.id}"


def _parse_cursor(cursor: str | None) -> tuple[datetime, str] | None:
    if not cursor:
        return None
    ts, _, cid = cursor.partition("|")
    try:
        return datetime.fromisoformat(ts), cid
    except ValueError:
        raise ValueError("invalid cursor") from None


# ---------------------------------------------------------------------------------- memory
class InMemoryConversationStore:
    def __init__(self) -> None:
        self._convs: dict[str, ConversationInfo] = {}
        self._msgs: dict[str, list[StoredMessage]] = {}
        self._keys: dict[tuple[str, str], int] = {}
        self._turns: dict[int, dict] = {}  # message id -> state, attempts, owner, heartbeat_at, error
        self._ids = itertools.count(1)
        self._lock = asyncio.Lock()

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    async def claim(self, conversation_id: str, user_id: str) -> bool:
        async with self._lock:
            conv = self._convs.get(conversation_id)
            if conv is None:
                now = self._now()
                self._convs[conversation_id] = ConversationInfo(conversation_id, user_id, "bot", None, None, now, now)
                self._msgs[conversation_id] = []
                return True
            return conv.user_id == user_id

    async def owner(self, conversation_id: str) -> str | None:
        conv = self._convs.get(conversation_id)
        return conv.user_id if conv else None

    def _append_locked(self, conversation_id, role, content, status, metadata, idempotency_key) -> int | None:
        if conversation_id not in self._convs:
            raise KeyError(f"unknown conversation {conversation_id}")  # same as the FK in Postgres
        if idempotency_key and (conversation_id, idempotency_key) in self._keys:
            return None
        msg = StoredMessage(next(self._ids), conversation_id, role, content, status, dict(metadata or {}), self._now())
        self._msgs[conversation_id].append(msg)
        if idempotency_key:
            self._keys[(conversation_id, idempotency_key)] = msg.id
        conv = self._convs[conversation_id]
        title = conv.title or (content[:TITLE_CHARS] if role == "user" and content else None)
        self._convs[conversation_id] = replace(conv, title=title, updated_at=msg.created_at)
        return msg.id

    async def append(self, conversation_id, role, content, *, status=None, metadata=None, idempotency_key=None):
        async with self._lock:
            return self._append_locked(conversation_id, role, content, status, metadata, idempotency_key)

    def _with_state(self, m: StoredMessage) -> StoredMessage:
        t = self._turns.get(m.id)
        return replace(m, state=t["state"]) if t else m

    async def find(self, conversation_id, idempotency_key):
        mid = self._keys.get((conversation_id, idempotency_key))
        found = next((m for m in self._msgs.get(conversation_id, []) if m.id == mid), None)
        return self._with_state(found) if found else None

    # -- save-first turns ------------------------------------------------------------
    async def begin_turn(self, conversation_id, content, *, owner, metadata=None, idempotency_key=None,
                         max_attempts=3):
        async with self._lock:
            mid = self._append_locked(conversation_id, "user", content, None, metadata, idempotency_key)
            if mid is not None:
                self._turns[mid] = {"state": "processing", "attempts": 1, "owner": owner,
                                    "heartbeat_at": self._now(), "error": None}
                return TurnClaim("claimed", mid, 1)
            mid = self._keys[(conversation_id, idempotency_key)]
            t = self._turns.get(mid)
            if t is None or t["state"] == "done":
                return TurnClaim("done", mid)  # untracked row (e.g. forwarded to an agent) counts as handled
            if t["state"] == "processing":
                return TurnClaim("processing", mid, t["attempts"])
            if t["attempts"] >= max_attempts:
                return TurnClaim("gave_up", mid, t["attempts"])
            t.update(state="processing", attempts=t["attempts"] + 1, owner=owner, heartbeat_at=self._now(), error=None)
            return TurnClaim("claimed", mid, t["attempts"])

    def _owned(self, message_id, owner, attempt) -> dict | None:
        t = self._turns.get(message_id)
        ok = t and t["state"] == "processing" and t["owner"] == owner and t["attempts"] == attempt
        return t if ok else None

    async def heartbeat(self, message_id, owner, attempt):
        async with self._lock:
            if t := self._owned(message_id, owner, attempt):
                t["heartbeat_at"] = self._now()
                return True
            return False

    async def finish_turn(self, message_id, owner, attempt, *, failed=False, error=None):
        async with self._lock:
            if t := self._owned(message_id, owner, attempt):
                t.update(state="failed" if failed else "done", error=(error or "")[:ERROR_CHARS] or None)
                return True
            return False

    async def release_owner(self, owner):
        async with self._lock:
            mine = [t for t in self._turns.values() if t["state"] == "processing" and t["owner"] == owner]
            for t in mine:
                t.update(state="failed", error="worker shut down")
            return len(mine)

    async def sweep_stuck(self, stale_after_s, limit=1000):
        async with self._lock:
            cutoff = self._now() - timedelta(seconds=stale_after_s)
            stuck = [t for t in self._turns.values() if t["state"] == "processing" and t["heartbeat_at"] < cutoff][:limit]
            for t in stuck:
                t.update(state="failed", error="stale: no heartbeat")
            return len(stuck)

    async def set_status(self, conversation_id, status, handoff_ticket=None):
        conv = self._convs.get(conversation_id)
        if conv:
            self._convs[conversation_id] = replace(conv, status=status, updated_at=self._now(),
                                                   handoff_ticket=handoff_ticket or conv.handoff_ticket)

    async def messages(self, conversation_id, *, before_id=None, limit=50):
        msgs = [m for m in self._msgs.get(conversation_id, []) if before_id is None or m.id < before_id]
        return [self._with_state(m) for m in msgs[-limit:]]

    async def conversations(self, user_id, *, cursor=None, limit=20):
        after = _parse_cursor(cursor)
        convs = sorted((c for c in self._convs.values() if c.user_id == user_id),
                       key=lambda c: (c.updated_at, c.id), reverse=True)
        if after:
            convs = [c for c in convs if (c.updated_at, c.id) < after]
        page = convs[:limit]
        return page, (_cursor(page[-1]) if len(convs) > limit else None)

    async def delete(self, conversation_id):
        async with self._lock:
            existed = self._convs.pop(conversation_id, None) is not None
            gone = {m.id for m in self._msgs.pop(conversation_id, [])}
            self._keys = {k: v for k, v in self._keys.items() if k[0] != conversation_id}
            self._turns = {k: v for k, v in self._turns.items() if k not in gone}
            return existed

    async def purge(self, older_than_days):
        cutoff = self._now() - timedelta(days=older_than_days)
        old = [cid for cid, c in self._convs.items() if c.updated_at < cutoff]
        for cid in old:
            await self.delete(cid)
        return len(old)

    async def ping(self):
        return True

    async def close(self):
        pass


# -------------------------------------------------------------------------------- postgres
REQUIRED_SCHEMA_VERSION = 2  # 0002: save-first turns (state / attempts / owner / heartbeat)


class PostgresConversationStore:
    def __init__(self, url: str, min_size: int = 1, max_size: int = 10, pool=None) -> None:
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool

        self.url = url
        self.pool = pool or AsyncConnectionPool(
            url, min_size=min_size, max_size=max_size, open=False, timeout=5,
            kwargs={"row_factory": dict_row, "autocommit": True},
            name="conversations",
        )
        self._opened = False
        self._open_lock = asyncio.Lock()

    async def _ensure_open(self) -> None:
        if not self._opened:
            async with self._open_lock:
                if not self._opened:
                    await self.pool.open(wait=False)
                    self._opened = True

    async def _fetch(self, sql: str, params: tuple = ()) -> list[dict]:
        await self._ensure_open()
        async with self.pool.connection() as conn:
            cur = await conn.execute(sql, params)
            return await cur.fetchall() if cur.description else []

    # -- ownership --------------------------------------------------------------------
    async def claim(self, conversation_id: str, user_id: str) -> bool:
        rows = await self._fetch(
            """
            WITH ins AS (
                INSERT INTO conversations (id, user_id) VALUES (%s, %s)
                ON CONFLICT (id) DO NOTHING
                RETURNING user_id
            )
            SELECT user_id FROM ins
            UNION ALL
            SELECT user_id FROM conversations WHERE id = %s
            LIMIT 1
            """,
            (conversation_id, user_id, conversation_id),
        )
        return bool(rows) and rows[0]["user_id"] == user_id

    async def owner(self, conversation_id: str) -> str | None:
        rows = await self._fetch("SELECT user_id FROM conversations WHERE id = %s", (conversation_id,))
        return rows[0]["user_id"] if rows else None

    # -- messages ---------------------------------------------------------------------
    async def append(self, conversation_id, role, content, *, status=None, metadata=None, idempotency_key=None):
        await self._ensure_open()
        async with self.pool.connection() as conn, conn.transaction():
            cur = await conn.execute(
                """
                INSERT INTO messages (conversation_id, role, content, status, metadata, idempotency_key)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT ON CONSTRAINT messages_idempotent DO NOTHING
                RETURNING id
                """,
                (conversation_id, role, content, status, json.dumps(metadata or {}, default=str), idempotency_key),
            )
            row = await cur.fetchone()
            if row is None:
                return None  # duplicate delivery
            await self._touch(conn, conversation_id, role, content)
            return row["id"]

    @staticmethod
    async def _touch(conn, conversation_id: str, role: str, content: str) -> None:
        await conn.execute(
            """
            UPDATE conversations
               SET updated_at = now(),
                   title = COALESCE(title, CASE WHEN %s = 'user' AND %s <> '' THEN left(%s, %s) END)
             WHERE id = %s
            """,
            (role, content, content, TITLE_CHARS, conversation_id),
        )

    @staticmethod
    def _msg(r: dict) -> StoredMessage:
        return StoredMessage(r["id"], r["conversation_id"], r["role"], r["content"], r["status"],
                             r["metadata"] or {}, r["created_at"], r.get("state"))

    # -- save-first turns -------------------------------------------------------------
    async def begin_turn(self, conversation_id, content, *, owner, metadata=None, idempotency_key=None,
                         max_attempts=3):
        await self._ensure_open()
        async with self.pool.connection() as conn, conn.transaction():
            # 1. the insert IS the lock: only one request gets a row back for this message id
            cur = await conn.execute(
                """
                INSERT INTO messages (conversation_id, role, content, metadata, idempotency_key,
                                      state, owner, heartbeat_at)
                VALUES (%s, 'user', %s, %s::jsonb, %s, 'processing', %s, now())
                ON CONFLICT ON CONSTRAINT messages_idempotent DO NOTHING
                RETURNING id, attempts
                """,
                (conversation_id, content, json.dumps(metadata or {}, default=str), idempotency_key, owner),
            )
            if row := await cur.fetchone():
                await self._touch(conn, conversation_id, "user", content)
                return TurnClaim("claimed", row["id"], row["attempts"])
            # 2. duplicate: re-claim it if a previous attempt failed (atomic: one retry wins)
            cur = await conn.execute(
                """
                UPDATE messages
                   SET state = 'processing', attempts = attempts + 1, owner = %s,
                       heartbeat_at = now(), error = NULL
                 WHERE conversation_id = %s AND idempotency_key = %s
                   AND state = 'failed' AND attempts < %s
                RETURNING id, attempts
                """,
                (owner, conversation_id, idempotency_key, max_attempts),
            )
            if row := await cur.fetchone():
                return TurnClaim("claimed", row["id"], row["attempts"])
            # 3. otherwise report what the row is doing
            cur = await conn.execute(
                "SELECT id, state, attempts FROM messages WHERE conversation_id = %s AND idempotency_key = %s",
                (conversation_id, idempotency_key),
            )
            row = await cur.fetchone()
            if row["state"] == "processing":
                return TurnClaim("processing", row["id"], row["attempts"])
            if row["state"] == "failed":
                return TurnClaim("gave_up", row["id"], row["attempts"])
            return TurnClaim("done", row["id"], row["attempts"])  # done, or an untracked row

    async def heartbeat(self, message_id, owner, attempt):
        rows = await self._fetch(
            """
            UPDATE messages SET heartbeat_at = now()
             WHERE id = %s AND state = 'processing' AND owner = %s AND attempts = %s
            RETURNING id
            """,
            (message_id, owner, attempt),
        )
        return bool(rows)

    async def finish_turn(self, message_id, owner, attempt, *, failed=False, error=None):
        rows = await self._fetch(
            """
            UPDATE messages SET state = %s, error = %s
             WHERE id = %s AND state = 'processing' AND owner = %s AND attempts = %s
            RETURNING id
            """,
            ("failed" if failed else "done", (error or "")[:ERROR_CHARS] or None, message_id, owner, attempt),
        )
        return bool(rows)

    async def release_owner(self, owner):
        rows = await self._fetch(
            """
            UPDATE messages SET state = 'failed', error = 'worker shut down'
             WHERE state = 'processing' AND owner = %s
            RETURNING id
            """,
            (owner,),
        )
        return len(rows)

    async def sweep_stuck(self, stale_after_s, limit=1000):
        # SKIP LOCKED: overlapping sweeps (or a slow one) never block each other or a live turn.
        rows = await self._fetch(
            """
            UPDATE messages SET state = 'failed', error = 'stale: no heartbeat'
             WHERE id IN (
                   SELECT id FROM messages
                    WHERE state = 'processing'
                      AND heartbeat_at < now() - make_interval(secs => %s)
                    ORDER BY heartbeat_at
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED)
            RETURNING id
            """,
            (float(stale_after_s), limit),
        )
        return len(rows)

    async def find(self, conversation_id, idempotency_key):
        rows = await self._fetch(
            "SELECT * FROM messages WHERE conversation_id = %s AND idempotency_key = %s",
            (conversation_id, idempotency_key),
        )
        return self._msg(rows[0]) if rows else None

    async def set_status(self, conversation_id, status, handoff_ticket=None):
        await self._fetch(
            "UPDATE conversations SET status = %s, handoff_ticket = COALESCE(%s, handoff_ticket), updated_at = now() WHERE id = %s",
            (status, handoff_ticket, conversation_id),
        )

    async def messages(self, conversation_id, *, before_id=None, limit=50):
        rows = await self._fetch(
            """
            SELECT * FROM messages
             WHERE conversation_id = %s AND (%s::bigint IS NULL OR id < %s)
             ORDER BY id DESC LIMIT %s
            """,
            (conversation_id, before_id, before_id, limit),
        )
        return [self._msg(r) for r in reversed(rows)]  # chronological

    async def conversations(self, user_id, *, cursor=None, limit=20):
        after = _parse_cursor(cursor)
        rows = await self._fetch(
            """
            SELECT * FROM conversations
             WHERE user_id = %s AND (%s::timestamptz IS NULL OR (updated_at, id) < (%s, %s))
             ORDER BY updated_at DESC, id DESC
             LIMIT %s
            """,
            (user_id, after[0] if after else None, after[0] if after else None, after[1] if after else "", limit + 1),
        )
        convs = [ConversationInfo(r["id"], r["user_id"], r["status"], r["title"], r["handoff_ticket"],
                                  r["created_at"], r["updated_at"]) for r in rows]
        page = convs[:limit]
        return page, (_cursor(page[-1]) if len(convs) > limit else None)

    async def delete(self, conversation_id):
        rows = await self._fetch("DELETE FROM conversations WHERE id = %s RETURNING id", (conversation_id,))
        return bool(rows)

    async def purge(self, older_than_days):
        rows = await self._fetch(
            "DELETE FROM conversations WHERE updated_at < now() - make_interval(days => %s) RETURNING id",
            (older_than_days,),
        )
        return len(rows)

    async def ping(self):
        """Reachable AND migrated: a pod must not serve traffic against an old schema."""
        try:
            rows = await self._fetch("SELECT max(version) AS v FROM schema_migrations")
            return bool(rows) and (rows[0]["v"] or 0) >= REQUIRED_SCHEMA_VERSION
        except Exception:
            return False

    async def close(self):
        if self._opened:
            await self.pool.close()


def build_history(database_url: str | None, min_size: int = 1, max_size: int = 10) -> ConversationStore:
    if database_url:
        return PostgresConversationStore(database_url, min_size, max_size)
    return InMemoryConversationStore()
