"""Command line: `python -m northwind_rag.cli ingest | migrate | purge [--days N] | sweep [--stale-after S] | ask "question" [--stream] [--groups g1,g2]`."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from northwind_rag.config import get_settings
from northwind_rag.models import build_suite
from northwind_rag.pipeline import Pipeline
from northwind_rag.store import KnowledgeBase


def _ingest() -> None:
    settings = get_settings()
    kb = KnowledgeBase(settings, build_suite(settings).embeddings)
    report = kb.ingest()
    print(f"Ingested {report.files} files -> {report.chunks} chunks (removed {report.removed_stale} stale)")


async def _ask(question: str, stream: bool, groups: list[str]) -> None:
    p = Pipeline.create()
    if not stream:
        print(json.dumps({k: v for k, v in (await p.ask(question, groups=groups)).items() if k != "contexts"}, indent=2))
        return
    async for ev in p.stream(question, groups=groups):
        name, data = ev["event"], ev["data"]
        if name == "delta":
            sys.stdout.write(data["text"])
            sys.stdout.flush()
        elif name == "status":
            print(f"\033[2m[{data['stage']}]\033[0m", flush=True)
        elif name == "done":
            print("\n\033[2m" + json.dumps({k: v for k, v in data.items() if k != "contexts"}, indent=2) + "\033[0m")
        else:
            print(f"\n\033[1m<{name}>\033[0m {json.dumps(data, ensure_ascii=False)}")


def _migrate() -> None:
    from northwind_rag.migrate import migrate

    settings = get_settings()
    if not settings.database_url:
        raise SystemExit("RAG_DATABASE_URL is not set")
    applied = migrate(settings.database_url)
    print(f"applied: {', '.join(applied)}" if applied else "schema up to date")


async def _purge(days: int | None) -> None:
    from northwind_rag.history import build_history

    settings = get_settings()
    store = build_history(settings.database_url)
    try:
        n = await store.purge(days or settings.history_retention_days)
        print(f"purged {n} conversation(s) idle for more than {days or settings.history_retention_days} days")
    finally:
        await store.close()


async def _sweep(stale_after_s: float | None) -> None:
    """Fail questions stuck in 'processing' (their pod died), so a retry can answer them."""
    from northwind_rag.history import build_history

    settings = get_settings()
    if not settings.database_url:
        raise SystemExit("RAG_DATABASE_URL is not set")
    stale = stale_after_s or settings.turn_stale_after_s
    store = build_history(settings.database_url)
    try:
        n = await store.sweep_stuck(stale)
        print(f"swept {n} question(s) with no heartbeat for {stale:g}s")
    finally:
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="northwind-rag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="(re)index data/docs into Chroma and the lexical index")
    sub.add_parser("migrate", help="apply pending PostgreSQL migrations")
    purge = sub.add_parser("purge", help="delete conversations idle longer than the retention period")
    purge.add_argument("--days", type=int, default=None)
    sweep = sub.add_parser("sweep", help="fail questions stuck in 'processing' after a pod crash")
    sweep.add_argument("--stale-after", type=float, default=None, help="seconds without heartbeat")
    ask = sub.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    ask.add_argument("--stream", action="store_true")
    ask.add_argument("--groups", default="", help="comma-separated groups, e.g. it-staff (default: public only)")
    args = parser.parse_args()
    if args.cmd == "ingest":
        _ingest()
    elif args.cmd == "migrate":
        _migrate()
    elif args.cmd == "purge":
        asyncio.run(_purge(args.days))
    elif args.cmd == "sweep":
        asyncio.run(_sweep(args.stale_after))
    else:
        asyncio.run(_ask(args.question, args.stream, [g for g in args.groups.split(",") if g.strip()]))


if __name__ == "__main__":
    main()
