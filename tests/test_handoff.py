import httpx
import pytest

from northwind_rag.api import app
from northwind_rag.config import Settings
from northwind_rag.graph import build_graph
from northwind_rag.guardrails.patterns import wants_live_agent
from northwind_rag.handoff import HandoffRequest, HandoffResult, LiveAgentService, StubLiveAgentService
from northwind_rag.pipeline import Pipeline
from tests.test_streaming_api import stream


@pytest.fixture
def with_agent(pipeline):
    """Pipeline sharing the KB, with its own live agent service and optional settings overrides."""

    def make(agent: LiveAgentService, **overrides) -> Pipeline:
        settings = pipeline.settings.model_copy(update=overrides)
        graph = build_graph(pipeline.suite, pipeline.kb, settings, agent)
        return Pipeline(settings, pipeline.suite, pipeline.kb, graph, agent)

    return make


@pytest.mark.parametrize("text", [
    "Can I talk to a real person?",
    "I want a human agent",
    "please connect me with the helpdesk",
    "escalate this",
    "Is there live support available?",
])
def test_agent_request_detected(text):
    assert wants_live_agent(text)


@pytest.mark.parametrize("text", [
    "How do I contact the helpdesk about VPN?",
    "What is the helpdesk phone extension?",
    "How often do engineers get a new laptop?",
])
def test_normal_questions_not_detected(text):
    assert not wants_live_agent(text)


def test_stub_satisfies_protocol():
    assert isinstance(StubLiveAgentService(), LiveAgentService)


async def test_user_request_forwarded_with_sanitized_question(with_agent):
    agent = StubLiveAgentService()
    out = await with_agent(agent).ask(
        "I want a human agent, my card 4111 1111 1111 1111 was charged twice", conversation_id="c-42"
    )
    assert out["status"] == "handed_off" and out["attempts"] == 0  # never generated an answer
    req = agent.requests[-1]
    assert req.reason == "user_requested" and req.conversation_id == "c-42"
    assert "4111" not in req.question and req.input_pii_redacted == ["card"]
    assert out["handoff"]["ticket_id"] in out["answer"]


async def test_no_answer_is_forwarded(with_agent):
    agent = StubLiveAgentService()
    out = await with_agent(agent).ask("Does Northwind reimburse gym memberships?")
    assert out["status"] == "no_context"
    assert agent.requests[-1].reason == "no_answer"
    assert out["answer"].startswith("I don't know") and out["handoff"]["ticket_id"] in out["answer"]


async def test_answered_question_is_not_forwarded(with_agent):
    agent = StubLiveAgentService()
    out = await with_agent(agent).ask("How often do engineers get a new laptop?")
    assert out["status"] == "answered" and out["handoff"] is None and not agent.requests


async def test_handoff_can_be_disabled(with_agent):
    agent = StubLiveAgentService()
    out = await with_agent(agent, handoff_on_no_answer=False).ask("Does Northwind reimburse gym memberships?")
    assert out["handoff"] is None and not agent.requests


async def test_broken_integration_never_breaks_chat(with_agent):
    class Broken:
        async def handoff(self, request: HandoffRequest) -> HandoffResult:
            raise ConnectionError("desk API down")

        async def is_available(self) -> bool:
            return False

    out = await with_agent(Broken()).ask("Can I talk to a real person?")
    assert out["handoff"]["status"] == "unavailable"
    assert "helpdesk@northwind.example" in out["answer"]


async def test_stream_emits_handoff_event(with_agent):
    events = await stream(with_agent(StubLiveAgentService()), "Does Northwind reimburse gym memberships?")
    names = [e for e, _ in events]
    assert names.index("final") < names.index("handoff") < names.index("done")
    handoff = next(d for e, d in events if e == "handoff")
    assert handoff["reason"] == "no_answer" and handoff["ticket_id"]


async def test_handoff_endpoint_redacts(pipeline):
    app.state.pipeline = pipeline
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/handoff", json={"question": "mail me at maria@gmail.com", "conversation_id": "c-1"})
    assert r.status_code == 200 and r.json()["ticket_id"]
    assert "maria@gmail.com" not in pipeline.live_agent.requests[-1].question
