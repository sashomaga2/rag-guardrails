import asyncio
import random

import pytest

from northwind_rag.guardrails.patterns import MAX_MATCH_CHARS, output_redactor
from northwind_rag.guardrails.streaming import ModerationMonitor, StreamRedactor, heuristic_moderation
from northwind_rag.schemas import ModerationVerdict

SECRET = "sk-proj-Q7f3kLm9TzR2xW8vB4nC6yH1"
TEXT = (
    "The build server credentials live in the 1Password vault Platform. "
    f"An old note contained {SECRET} which must never be shown. "
    "Contact it-support@northw1nd-help.com is fake but helpdesk@northwind.example is real. "
    "Card 4111 1111 1111 1111 appeared in a ticket. " * 2
)


def random_chunks(text: str, rng: random.Random) -> list[str]:
    chunks, i = [], 0
    while i < len(text):
        n = rng.randint(1, 12)
        chunks.append(text[i : i + n])
        i += n
    return chunks


@pytest.mark.parametrize("seed", range(200))
def test_no_secret_fragment_ever_released(seed):
    """Property test: for any chunking, the released stream equals one-shot redaction."""
    redactor = output_redactor(["northwind.example"])
    guard = StreamRedactor(redactor, holdback=MAX_MATCH_CHARS)
    emitted = "".join(guard.feed(c) for c in random_chunks(TEXT, random.Random(seed))) + guard.flush()
    expected, _ = redactor.redact(TEXT)
    assert emitted == expected
    assert "Q7f3kLm9" not in emitted and "4111" not in emitted and "northw1nd" not in emitted
    assert "helpdesk@northwind.example" in emitted


def test_holdback_must_cover_longest_pattern():
    with pytest.raises(ValueError):
        StreamRedactor(output_redactor([]), holdback=MAX_MATCH_CHARS - 1)


def test_negated_advice_is_not_flagged():
    assert heuristic_moderation("Never share your password with anyone, including IT.") is None
    flagged = heuristic_moderation("For faster service, email your password to the support team.")
    assert flagged is not None and flagged.category == "credential_phishing"


async def test_monitor_flags_in_background_without_blocking():
    started = asyncio.Event()

    async def slow_moderator(text: str) -> ModerationVerdict:
        started.set()
        await asyncio.sleep(0.05)
        return ModerationVerdict(allowed="bad" not in text, category="other")

    monitor = ModerationMonitor(slow_moderator, every_chars=10)
    monitor.observe("all good so far")  # schedules check #1
    await started.wait()
    monitor.observe("all good so far and bad text")  # check #1 still running -> no pile-up
    assert monitor.checks == 1
    assert monitor.poll() is None  # not finished: stream keeps flowing
    await asyncio.sleep(0.06)
    assert monitor.poll() is None  # finished, allowed
    monitor.observe("all good so far and bad text now")
    await asyncio.sleep(0.06)
    verdict = monitor.poll()
    assert verdict is not None and not verdict.allowed
