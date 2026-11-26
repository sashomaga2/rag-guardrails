"""Streaming output guardrails.

StreamRedactor   - blocking, deterministic. Holds back the last N chars so a secret split
                   across chunks ("sk-proj-Q7f" + "3kLm9...") is never partly released.
ModerationMonitor - non-blocking. Runs a classifier on the cumulative text every N chars in a
                   background task; if it flags the text, the caller retracts what was shown.
"""

from __future__ import annotations

import asyncio
import re
from typing import Awaitable, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from northwind_rag.guardrails.patterns import MAX_MATCH_CHARS, Redactor
from northwind_rag.models import ModelSuite
from northwind_rag.prompts import MODERATION_SYSTEM
from northwind_rag.schemas import ModerationVerdict


class StreamRedactor:
    """Feed raw deltas in, get safe text out.

    Correctness argument: we only release text up to `cut = len(buffer) - holdback`, moved left
    so it never splits a current match. A future match that started before `cut` would have to be
    longer than `holdback` >= MAX_MATCH_CHARS, which no pattern allows.
    """

    def __init__(self, redactor: Redactor, holdback: int = 128):
        if holdback < MAX_MATCH_CHARS:
            raise ValueError("holdback must be >= MAX_MATCH_CHARS")
        self.redactor = redactor
        self.holdback = holdback
        self.reset()

    def reset(self) -> None:
        self._buffer = ""
        self.released = ""  # everything the user has seen so far (already redacted)
        self.findings: list[str] = []

    def _emit(self, text: str) -> str:
        safe, found = self.redactor.redact(text)
        self.findings.extend(found)
        self.released += safe
        return safe

    def feed(self, delta: str) -> str:
        self._buffer += delta
        if len(self._buffer) <= self.holdback:
            return ""
        cut = len(self._buffer) - self.holdback
        for f in self.redactor.find(self._buffer):
            if f.start < cut < f.end:
                cut = f.start
        if cut <= 0:
            return ""
        out, self._buffer = self._buffer[:cut], self._buffer[cut:]
        return self._emit(out)

    def flush(self) -> str:
        out, self._buffer = self._buffer, ""
        return self._emit(out) if out else ""


Moderator = Callable[[str], Awaitable[ModerationVerdict]]


class ModerationMonitor:
    """Schedules moderation of the cumulative text without blocking the token stream."""

    def __init__(self, moderate: Moderator, every_chars: int = 240):
        self.moderate = moderate
        self.every_chars = every_chars
        self.reset()

    def reset(self) -> None:
        self._task: asyncio.Task[ModerationVerdict] | None = None
        self._checked_upto = 0
        self.checks = 0

    def observe(self, text: str) -> None:
        """Start a check if enough new text arrived and no check is running (no pile-up)."""
        if len(text) - self._checked_upto < self.every_chars:
            return
        if self._task is not None and not self._task.done():
            return
        self._checked_upto = len(text)
        self.checks += 1
        self._task = asyncio.create_task(self.moderate(text))

    def poll(self) -> ModerationVerdict | None:
        """Return a blocking verdict if a finished check flagged the text."""
        if self._task is not None and self._task.done():
            task, self._task = self._task, None
            verdict = task.result()
            if not verdict.allowed:
                return verdict
        return None

    async def final(self, text: str) -> ModerationVerdict:
        """At end of stream: wait for any running check, then check the complete text once."""
        if self._task is not None:
            verdict = await self._task
            self._task = None
            if not verdict.allowed:
                return verdict
        self.checks += 1
        return await self.moderate(text)

    def cancel(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()


# Offline stand-in for the moderation model: phrases a helpdesk answer must never contain.
_OFFLINE_BLOCK = (
    (re.compile(r"\b(?:send|email|e-mail|give|tell|share)\b[^.\n]{0,40}\b(?:your|the)\s+(?:password|mfa|otp|verification)\b", re.I), "credential_phishing"),
    (re.compile(r"\b(?:disable|turn off|bypass|uninstall)\b[^.\n]{0,30}\b(?:antivirus|defender|edr|mfa|firewall)\b", re.I), "security_bypass"),
)


_NEGATION = re.compile(r"\b(?:never|not|no one|nobody)\b|n't\b", re.I)


def heuristic_moderation(text: str) -> ModerationVerdict | None:
    """Sentence-level patterns; a negated sentence ("never share your password") is fine."""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if _NEGATION.search(sentence):
            continue
        for pattern, category in _OFFLINE_BLOCK:
            if pattern.search(sentence):
                return ModerationVerdict(allowed=False, category=category, reason="matched block pattern")
    return None


def make_moderator(suite: ModelSuite) -> Moderator:
    @traceable(name="stream_moderation", run_type="chain")
    async def moderate(text: str) -> ModerationVerdict:
        flagged = heuristic_moderation(text)  # cheap check first, also used online
        if flagged:
            return flagged
        if suite.guard is None:
            return ModerationVerdict(allowed=True)
        classifier = suite.guard.with_structured_output(ModerationVerdict)
        messages = [SystemMessage(MODERATION_SYSTEM), HumanMessage(f"<assistant_text>\n{text}\n</assistant_text>")]
        try:
            return await suite.call("guard", lambda: classifier.ainvoke(messages))
        except Exception as exc:  # fail closed: text we could not check is not shown
            return ModerationVerdict(allowed=False, category="other", reason=f"moderation unavailable: {type(exc).__name__}")

    return moderate
