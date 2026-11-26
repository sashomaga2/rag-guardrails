"""Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, SIGIR 2009).

    rrf(d) = Σ_lists  weight_list / (k + rank_list(d))       rank starts at 1

Why RRF for hybrid search: BM25 scores (unbounded, query-dependent) and cosine similarities
(0..1) live on different scales, so adding or normalising them is fragile. RRF uses only
ranks: a document near the top of either list scores well, and one near the top of both
scores best. k (default 60) damps the advantage of the very first ranks.

Example with k=60, both weights 1:
    doc A: dense #1, lexical #3  -> 1/61 + 1/63 = 0.03227
    doc B: dense #2, absent      -> 1/62        = 0.01613
    doc C: absent,   lexical #1  -> 1/61        = 0.01639
    order: A, C, B
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Sequence, TypeVar

T = TypeVar("T")


@dataclass
class Fused:
    item: object
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # list name -> 1-based rank


def rrf_merge(
    ranked_lists: dict[str, Sequence[T]],
    key=lambda x: x,
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[Fused]:
    """Merge ranked lists (best first) into one ranking. Duplicates within a list keep their
    best rank. Ties are broken by the best single rank, then by list order (deterministic)."""
    if k < 1:
        raise ValueError("k must be >= 1")
    weights = weights or {}
    acc: dict[Hashable, Fused] = {}
    for name, items in ranked_lists.items():
        w = weights.get(name, 1.0)
        seen: set[Hashable] = set()
        rank = 0
        for item in items:
            ident = key(item)
            if ident in seen:
                continue  # a duplicate must not consume a rank and push later items down
            seen.add(ident)
            rank += 1
            entry = acc.setdefault(ident, Fused(item=item, score=0.0))
            entry.score += w / (k + rank)
            entry.ranks[name] = rank
    return sorted(acc.values(), key=lambda f: (-f.score, min(f.ranks.values())))
