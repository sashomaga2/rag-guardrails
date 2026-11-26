"""Minimal, safe SQL migration runner.

- Migrations are numbered SQL files in northwind_rag/migrations (0001_*.sql, 0002_*.sql ...).
- Each file runs in its own transaction together with its row in schema_migrations, so a failed
  migration leaves nothing half-applied.
- A session-level advisory lock, taken before anything else, serialises runners: the Kubernetes
  Job, a developer laptop and a second Job started by a retry can't apply anything twice.
- Applied files are checksummed; editing a migration after it ran is an error, not a silent drift.

    python -m northwind_rag.cli migrate
"""

from __future__ import annotations

import hashlib
import logging
from importlib import resources

log = logging.getLogger(__name__)

LOCK_ID = 0x4E57_5241_4721  # "NWRAG!": constant advisory lock key for this app


def migration_files() -> list[tuple[int, str, str]]:
    files = sorted(p for p in resources.files("northwind_rag.migrations").iterdir() if p.name.endswith(".sql"))
    out = []
    for f in files:
        version = int(f.name.split("_", 1)[0])
        out.append((version, f.name, f.read_text(encoding="utf-8")))
    versions = [v for v, _, _ in out]
    if versions != sorted(set(versions)):
        raise RuntimeError(f"duplicate or unordered migration versions: {versions}")
    return out


def migrate(database_url: str) -> list[str]:
    """Apply pending migrations. Returns the names of the files applied by this call."""
    import psycopg

    applied_now: list[str] = []
    with psycopg.connect(database_url, autocommit=True) as conn:
        # Session-level lock taken FIRST: even `CREATE TABLE IF NOT EXISTS` races in PostgreSQL
        # (two sessions can both decide the table is missing and collide in pg_type).
        conn.execute("SELECT pg_advisory_lock(%s)", (LOCK_ID,))
        try:
            _apply(conn, applied_now)
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (LOCK_ID,))
    return applied_now


def _apply(conn, applied_now: list[str]) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version    integer     PRIMARY KEY,
            name       text        NOT NULL,
            checksum   text        NOT NULL,
            applied_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    for version, name, sql in migration_files():
        checksum = hashlib.sha256(sql.encode()).hexdigest()
        with conn.transaction():
            row = conn.execute("SELECT checksum FROM schema_migrations WHERE version = %s", (version,)).fetchone()
            if row:
                if row[0] != checksum:
                    raise RuntimeError(f"migration {name} was changed after it was applied; add a new migration instead")
                continue
            conn.execute(sql)
            conn.execute("INSERT INTO schema_migrations (version, name, checksum) VALUES (%s, %s, %s)",
                         (version, name, checksum))
            applied_now.append(name)
            log.info("applied migration %s", name)
