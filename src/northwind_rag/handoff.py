"""Live-agent handoff: the contract between the assistant and a human support desk.

Only the interface and a stub exist. To integrate a real desk (ServiceNow, Zendesk,
Genesys, Salesforce Service Cloud, a Teams queue...), implement `LiveAgentService`
and pass it to `Pipeline.create(live_agent=...)`. Nothing else changes.

Privacy: the request carries the *sanitized* question only. Raw user input with PII
never leaves the assistant, not even to the human desk.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Literal, Protocol, runtime_checkable

log = logging.getLogger(__name__)

HandoffReason = Literal["user_requested", "no_answer", "unverified_answer"]
HandoffStatus = Literal["queued", "agent_assigned", "unavailable", "not_implemented"]


@dataclass(frozen=True)
class HandoffRequest:
    """Everything a human agent needs to pick up the conversation without re-asking."""

    reason: HandoffReason
    question: str  # sanitized (PII-redacted) question
    conversation_id: str | None = None
    user_id: str | None = None
    retrieved_sources: list[str] = field(default_factory=list)
    draft_answer: str | None = None  # what the assistant produced, if anything
    unsupported_claims: list[str] = field(default_factory=list)
    input_pii_redacted: list[str] = field(default_factory=list)  # kinds only, never values
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class HandoffResult:
    status: HandoffStatus
    ticket_id: str | None = None
    queue: str | None = None
    estimated_wait_minutes: int | None = None
    message: str = ""  # shown to the user

    def to_dict(self) -> dict:
        return asdict(self)


@runtime_checkable
class LiveAgentService(Protocol):
    """Forward a conversation to a human agent."""

    async def handoff(self, request: HandoffRequest) -> HandoffResult:
        """Create a ticket / chat session for a human agent and return how it went.

        Implementations must not raise for expected conditions (desk closed, queue full):
        return status="unavailable" with a user-facing message instead. Raise only for
        bugs; the graph turns unexpected errors into "unavailable".
        """
        ...

    async def is_available(self) -> bool:
        """Whether agents are online right now (business hours, staffing)."""
        ...

    async def send_user_message(self, conversation_id: str, text: str) -> None:
        """Forward a follow-up message from the user while a human owns the conversation.

        The agent's replies come back the other way: the desk calls POST /agent/webhook and
        the ConversationBus delivers them to whichever pod holds the user's connection.
        """
        ...


class StubLiveAgentService:
    """Placeholder until a real desk is integrated. Logs the request, does not forward it.

    Returns status="not_implemented" so the UI, the evals and the logs can tell a real
    handoff from a pretend one. Records requests in memory for tests and local demos.
    """

    def __init__(self) -> None:
        self.requests: list[HandoffRequest] = []
        self.user_messages: list[tuple[str, str]] = []

    async def handoff(self, request: HandoffRequest) -> HandoffResult:
        self.requests.append(request)
        ticket_id = f"STUB-{uuid.uuid4().hex[:8].upper()}"
        log.warning("live agent handoff NOT forwarded (stub): %s reason=%s", ticket_id, request.reason)
        # TODO: replace with a real integration, e.g.
        #   ServiceNow: POST /api/now/table/incident  (short_description=question, work_notes=context)
        #   Zendesk:    POST /api/v2/tickets           (ticket.comment.body=question)
        #   Live chat:  open a session in the contact-center API and return its queue position
        return HandoffResult(
            status="not_implemented",
            ticket_id=ticket_id,
            queue="it-helpdesk",
            message=(
                "I've passed your question to the IT helpdesk team. "
                f"Your reference is {ticket_id}. You can also email helpdesk@northwind.example "
                "or call extension 4357."
            ),
        )

    async def is_available(self) -> bool:
        return True

    async def send_user_message(self, conversation_id: str, text: str) -> None:
        self.user_messages.append((conversation_id, text))
        log.warning("user message for %s NOT forwarded (stub)", conversation_id)
        # TODO: e.g. ServiceNow: POST /api/now/table/incident/{sys_id} with comments=text


HANDOFF_UNAVAILABLE = HandoffResult(
    status="unavailable",
    message="I couldn't reach the IT helpdesk right now. Please email helpdesk@northwind.example or call extension 4357.",
)
