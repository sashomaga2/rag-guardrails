"""Tiny lexical helpers used by the offline models and the offline grounding check."""

from __future__ import annotations

import re

CITATION_RE = re.compile(r"\[([A-Za-z0-9_.\-]+#\d+)\]")
_WORD_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    """a an the and or but if of to in on at by for with from as is are was were be been being do does did
    i you your yours we our it its this that these those my me can could should would will shall may might
    must how what when where which who whom why there their them they he she his her not no so than then too
    very just about into over under after before again more most such only own same any all each other some
    have has had having get got also please tell know need want""".split()
)


def stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def content_tokens(text: str) -> list[str]:
    return [stem(w) for w in _WORD_RE.findall(text.lower()) if w not in STOPWORDS]


def strip_citations(text: str) -> str:
    return CITATION_RE.sub("", text).strip()


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*").strip()
        if not line or line.startswith("#"):
            continue
        sentences.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip())
    return sentences
