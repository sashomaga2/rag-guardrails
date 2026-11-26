"""Document-level access control for retrieval.

Who may read a document is declared in the document itself, in a front-matter block:

    ---
    access: it-staff, security
    ---
    # Break-glass admin access
    ...

No block, or `access: public`, means every user. Group names are lowercase [a-z0-9_-].

How it is enforced
------------------
1. Ingest stamps the document's groups on every chunk:
     Chroma         access_groups="it-staff,security" plus one boolean per group (acl_it-staff=True ...)
                    because Chroma metadata values are scalars: no list membership filter.
     Elasticsearch  access_groups: ["it-staff", "security"]   (keyword array)
2. Search FILTERS INSIDE THE QUERY with the user's groups + "public":
     Chroma         where={"$or": [{"acl_public": True}, {"acl_it-staff": True}]}
     Elasticsearch  bool.filter: terms access_groups ["public", "it-staff"]
     local BM25     skips chunks the user can't read before scoring
   Filtering before ranking (not after) matters twice: nothing restricted ever reaches the
   reranker, the LLM, a trace or a cache, and top-k is never "used up" by chunks that would be
   removed later, so the user still gets their best permitted chunks.
3. Defence in depth: after retrieval every chunk is checked again with `can_read`; a chunk that a
   store returned despite the filter is dropped and logged as an error.
Chunks ingested before this change carry no ACL fields, so they match no filter (fail closed)
until the next ingest stamps them.

The user's groups come from the verified JWT (Istio forwards its payload, see api.py); a request
with no groups sees public documents only.
"""

from __future__ import annotations

import re
from typing import Iterable

PUBLIC = "public"
ACL_PREFIX = "acl_"
_GROUP_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


def normalize_group(name: str) -> str | None:
    g = name.strip().lower()
    return g if _GROUP_RE.fullmatch(g) else None


def parse_groups(value: str | Iterable[str] | None, *, strict: bool) -> frozenset[str]:
    """Comma/space separated string or list -> set of valid group names.
    strict=True (ingest): an invalid name is an error. strict=False (tokens): it is ignored."""
    if value is None:
        return frozenset()
    items = re.split(r"[\s,]+", value) if isinstance(value, str) else [str(v) for v in value]
    out = set()
    for raw in filter(None, (i.strip() for i in items)):
        g = normalize_group(raw)
        if g is None:
            if strict:
                raise ValueError(f"invalid access group {raw!r}: use lowercase letters, digits, '_' or '-'")
            continue
        out.add(g)
    return frozenset(out)


def split_front_matter(text: str) -> tuple[dict[str, str], str]:
    """`---` key: value lines `---` at the very top -> (fields, body). Simple on purpose: only
    flat `key: value` lines, so a document can't smuggle in anything else."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {}, text
    fields: dict[str, str] = {}
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return fields, "".join(lines[i + 1:]).lstrip("\n")
        key, sep, value = line.partition(":")
        if not sep or not key.strip():
            raise ValueError(f"front matter line {i + 1} is not 'key: value': {line.strip()!r}")
        fields[key.strip().lower()] = value.strip()
    raise ValueError("front matter is not closed with '---'")


def document_groups(fields: dict[str, str]) -> frozenset[str]:
    groups = parse_groups(fields.get("access"), strict=True)
    if not groups or PUBLIC in groups:
        return frozenset({PUBLIC})  # public wins: "public, hr" is just public
    return groups


def acl_metadata(groups: frozenset[str]) -> dict:
    """Chunk metadata for both stores: a readable list + one boolean per group (Chroma filter)."""
    return {"access_groups": ",".join(sorted(groups)), **{f"{ACL_PREFIX}{g}": True for g in sorted(groups)}}


def readable_by(user_groups: Iterable[str]) -> list[str]:
    return sorted({PUBLIC, *user_groups})


def chroma_filter(user_groups: Iterable[str]) -> dict:
    clauses = [{f"{ACL_PREFIX}{g}": True} for g in readable_by(user_groups)]
    return clauses[0] if len(clauses) == 1 else {"$or": clauses}


def chunk_groups(metadata: dict) -> frozenset[str]:
    value = metadata.get("access_groups")
    if isinstance(value, (list, tuple)):
        return frozenset(value)
    return frozenset(filter(None, str(value or "").split(",")))


def can_read(metadata: dict, user_groups: Iterable[str]) -> bool:
    """A chunk with no ACL at all is NOT readable (fail closed)."""
    return bool(chunk_groups(metadata) & set(readable_by(user_groups)))
