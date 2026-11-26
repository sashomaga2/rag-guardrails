"""Deterministic detectors for secrets and PII, plus a redactor.

Every pattern has a bounded maximum length. The streaming guard relies on that:
if no match can be longer than MAX_MATCH_CHARS, holding back that many
characters guarantees a secret split across chunks is never partly released.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

MAX_MATCH_CHARS = 100  # upper bound of any match below; keep in sync when adding patterns


def _luhn_ok(candidate: str) -> bool:
    digits = [int(c) for c in candidate if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _phone_ok(candidate: str) -> bool:
    return 8 <= sum(c.isdigit() for c in candidate) <= 15


@dataclass(frozen=True)
class Detector:
    name: str
    category: str  # "secret" | "pii"
    regex: re.Pattern[str]
    validator: Callable[[str], bool] | None = None


DETECTORS: tuple[Detector, ...] = (
    # ---- secrets ------------------------------------------------------------
    Detector("api_key", "secret", re.compile(r"\bsk-(?:proj-|live-|test-)?[A-Za-z0-9_-]{16,48}\b")),
    Detector("aws_key", "secret", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    Detector("github_token", "secret", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36}\b")),
    Detector(
        "credential",
        "secret",
        # "password: hunter2", "api_key = abc..." -> only the value group is redacted
        re.compile(
            r"(?i)\b(?:password|passwd|pwd|secret|api[_-]?key|token)\s{0,3}[:=]\s{0,3}(?P<value>(?!\[REDACTED)[^\s'\"]{6,64})"
        ),
    ),
    # ---- PII ----------------------------------------------------------------
    Detector("email", "pii", re.compile(r"\b[A-Za-z0-9._%+-]{1,40}@[A-Za-z0-9.-]{1,40}\.[A-Za-z]{2,10}\b")),
    Detector("card", "pii", re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), _luhn_ok),
    Detector("iban", "pii", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b")),
    Detector("phone", "pii", re.compile(r"(?<!\w)\+\d{1,3}[ -]?\(?\d{1,4}\)?(?:[ -]?\d{2,4}){2,4}(?!\w)"), _phone_ok),
)


@dataclass(frozen=True)
class Finding:
    detector: str
    start: int
    end: int


@dataclass
class Redactor:
    """Finds and masks sensitive spans.

    detectors: which detector names to apply.
    allowed_email_domains: corporate addresses (helpdesk@...) are not PII here.
    """

    detectors: frozenset[str] = frozenset(d.name for d in DETECTORS)
    allowed_email_domains: tuple[str, ...] = ()
    _active: tuple[Detector, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._active = tuple(d for d in DETECTORS if d.name in self.detectors)
        self.allowed_email_domains = tuple(x.lower() for x in self.allowed_email_domains)

    def _allowed(self, det: Detector, value: str) -> bool:
        if det.name == "email":
            domain = value.rsplit("@", 1)[-1].lower()
            return any(domain == a or domain.endswith("." + a) for a in self.allowed_email_domains)
        return False

    def find(self, text: str) -> list[Finding]:
        raw: list[Finding] = []
        for det in self._active:
            for m in det.regex.finditer(text):
                start, end = m.span("value") if "value" in det.regex.groupindex else m.span()
                value = text[start:end]
                if det.validator and not det.validator(m.group()):
                    continue
                if self._allowed(det, value):
                    continue
                raw.append(Finding(det.name, start, end))
        # resolve overlaps: earliest first, longest wins
        raw.sort(key=lambda f: (f.start, -(f.end - f.start)))
        result: list[Finding] = []
        last_end = -1
        for f in raw:
            if f.start >= last_end:
                result.append(f)
                last_end = f.end
        return result

    def redact(self, text: str) -> tuple[str, list[str]]:
        findings = self.find(text)
        if not findings:
            return text, []
        out, pos = [], 0
        for f in findings:
            out.append(text[pos : f.start])
            out.append(f"[REDACTED:{f.detector}]")
            pos = f.end
        out.append(text[pos:])
        return "".join(out), [f.detector for f in findings]


SECRET_DETECTORS = frozenset(d.name for d in DETECTORS if d.category == "secret")
ALL_DETECTORS = frozenset(d.name for d in DETECTORS)


def input_redactor(allowed_email_domains: Iterable[str]) -> Redactor:
    """User input: strip all PII/secrets before it reaches LLMs, embeddings or logs."""
    return Redactor(ALL_DETECTORS, tuple(allowed_email_domains))


def context_redactor() -> Redactor:
    """Retrieved chunks: KB contacts are legitimate, but secrets/financial data are not."""
    return Redactor(SECRET_DETECTORS | {"card", "iban"})


def output_redactor(allowed_email_domains: Iterable[str]) -> Redactor:
    """Model output: secrets, financial data and non-corporate emails never reach the user."""
    return Redactor(SECRET_DETECTORS | {"card", "iban", "email"}, tuple(allowed_email_domains))


# ---------------------------------------------------------------------------
# Prompt-injection heuristics (cheap first line; the LLM classifier is second)
# ---------------------------------------------------------------------------
INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE | re.MULTILINE)
    for p in (
        r"\b(?:ignore|forget|override)\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier|preceding)\s+(?:instructions|prompts|rules|directions|messages)",
        r"\bdisregard\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|system|your)\b",
        r"\b(?:reveal|show|print|repeat|output|leak)\s+(?:me\s+)?(?:your|the)\s+(?:system\s+prompt|instructions|hidden\s+prompt|initial\s+prompt|prompt)",
        r"\byou\s+are\s+now\s+(?:a|an|in|the)\b",
        r"\b(?:developer|god|dan|jailbreak|unrestricted)\s+mode\b",
        r"<\|?(?:im_start|im_end|system|endoftext)\|?>",
        r"^\s*#{2,}\s*(?:system|instructions?)\b",
        r"\bact\s+as\s+(?:an?\s+)?(?:unfiltered|unrestricted|uncensored)",
        r"\bnote\s+to\s+(?:ai|llm|the)\s*(?:assistants?|models?|chatbots?)?\b",
    )
)


def injection_hits(text: str) -> list[str]:
    return [p.pattern for p in INJECTION_PATTERNS if p.search(text)]


# ---------------------------------------------------------------------------
# "I want a human" intent (cheap first line; the LLM classifier also has this label)
# ---------------------------------------------------------------------------
AGENT_REQUEST_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:talk|speak|chat|connect)\s+(?:me\s+)?(?:to|with)\s+(?:a\s+|an\s+|the\s+|someone\s+(?:from\s+)?(?:the\s+)?)?(?:real\s+|live\s+)?(?:human|person|agent|operator|technician|helpdesk|support|someone)\b",
        r"\b(?:live|human|real)\s+(?:agent|person|support|operator)\b",
        r"\b(?:escalate|transfer)\s+(?:this|me|my\s+(?:case|issue|ticket))\b",
        r"\b(?:i\s+(?:want|need)|get\s+me)\s+(?:a\s+|an\s+)?(?:human|person|agent|technician)\b",
    )
)


def wants_live_agent(text: str) -> bool:
    return any(p.search(text) for p in AGENT_REQUEST_PATTERNS)
