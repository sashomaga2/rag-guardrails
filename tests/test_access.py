"""Document-level access control: front matter, ACL stamping, filtering inside Chroma / BM25 /
Elasticsearch, the defence-in-depth post-check, JWT groups in the API, and end-to-end behaviour."""

import base64
import json
import logging

import httpx
import pytest
from langchain_core.documents import Document

from northwind_rag.access import (acl_metadata, can_read, chroma_filter, document_groups, parse_groups,
                                  readable_by, split_front_matter)
from northwind_rag.api import app, user_groups
from northwind_rag.config import Settings
from northwind_rag.lexical import ElasticsearchIndex, LocalBM25Index
from northwind_rag.models import HashingEmbeddings
from northwind_rag.store import KnowledgeBase
from tests.test_hybrid import FakeAsyncES, FakeES, StaticLexical, doc


def jwt_payload(claims: dict) -> str:
    """What Istio's outputPayloadToHeader sends: base64url(JSON), unpadded."""
    return base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")


# --------------------------------------------------------------------------- the ACL rules
def test_front_matter_is_parsed_and_removed_from_the_body():
    fields, body = split_front_matter("---\naccess: it-staff, security\nowner: it\n---\n\n# Title\ntext\n")
    assert fields == {"access": "it-staff, security", "owner": "it"} and body == "# Title\ntext\n"
    assert split_front_matter("# No front matter\n") == ({}, "# No front matter\n")


@pytest.mark.parametrize("text, error", [("---\naccess: hr\n", "not closed"),
                                         ("---\njust words\n---\n", "key: value")])
def test_malformed_front_matter_fails_the_ingest(text, error):
    with pytest.raises(ValueError, match=error):
        split_front_matter(text)


def test_document_groups():
    assert document_groups({}) == {"public"}  # no label: everyone
    assert document_groups({"access": "public, hr"}) == {"public"}  # public wins
    assert document_groups({"access": "IT-Staff,  security"}) == {"it-staff", "security"}
    with pytest.raises(ValueError, match="invalid access group"):
        document_groups({"access": "it staff!"})


def test_token_groups_ignore_invalid_names():
    assert parse_groups(["hr", "Not Valid!", "it-staff"], strict=False) == {"hr", "it-staff"}
    assert parse_groups("hr it-staff,finance", strict=False) == {"hr", "it-staff", "finance"}


def test_can_read_fails_closed():
    assert can_read(acl_metadata(frozenset({"public"})), [])
    restricted = acl_metadata(frozenset({"it-staff", "security"}))
    assert can_read(restricted, ["security"]) and not can_read(restricted, ["hr"]) and not can_read(restricted, [])
    assert not can_read({"chunk_id": "old#0"}, ["it-staff"])  # no label at all: nobody


def test_chroma_filter_shape():
    assert chroma_filter([]) == {"acl_public": True}
    assert chroma_filter(["hr", "it-staff"]) == {"$or": [{"acl_hr": True}, {"acl_it-staff": True}, {"acl_public": True}]}
    assert readable_by(["hr"]) == ["hr", "public"]


@pytest.mark.parametrize("header, expected", [
    (None, set()),
    (jwt_payload({"sub": "u1", "groups": ["it-staff", "hr"]}), {"it-staff", "hr"}),
    (jwt_payload({"sub": "u1", "groups": "it-staff hr"}), {"it-staff", "hr"}),
    (jwt_payload({"sub": "u1"}), set()),  # no claim: public only
    (jwt_payload({"groups": {"nested": "it-staff"}}), set()),  # wrong type
    (jwt_payload(["groups", "it-staff"]), set()),  # not an object
    ("%%% not base64 %%%", set()),
    (base64.urlsafe_b64encode(b"not json").decode(), set()),
])
def test_user_groups_from_the_verified_payload(header, expected):
    assert user_groups(header) == expected


# ------------------------------------------------------------- filtering inside the stores
@pytest.fixture
def kb(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "vpn.md").write_text("# VPN\n## Connect\nOpen GlobalProtect and connect to vpn.northwind.example.\n")
    (docs / "breakglass.md").write_text(
        "---\naccess: it-staff\n---\n# Break-glass\n## Approval\nThe duty manager approves break-glass admin access.\n")
    (docs / "payroll.md").write_text(
        "---\naccess: hr, finance\n---\n# Payroll\n## Calendar\nPayroll runs on the 25th; the payroll calendar is set by HR.\n")
    settings = Settings(offline=True, chroma_dir=tmp_path / "chroma", docs_dir=docs, reranker="none")
    kb = KnowledgeBase(settings, HashingEmbeddings())
    kb.ingest()
    return kb


def sources(docs: list[Document]) -> set[str]:
    return {d.metadata["source"] for d in docs}


async def test_restricted_document_is_invisible_without_the_group(kb):
    assert "breakglass.md" not in sources(await kb.search("break-glass admin access duty manager"))
    assert "breakglass.md" not in sources(await kb.search("break-glass admin access duty manager", groups={"hr"}))
    assert "breakglass.md" in sources(await kb.search("break-glass admin access duty manager", groups={"it-staff"}))


async def test_each_store_filters_on_its_own(kb):
    """Chroma and BM25 are checked separately: each must filter, not rely on the other."""
    q = "payroll calendar 25th"
    dense = await kb.vs.asimilarity_search_with_relevance_scores(q, k=10, filter={"acl_public": True})
    assert "payroll.md" not in {d.metadata["source"] for d, _ in dense}
    assert "payroll.md" not in {d.metadata["source"] for d, _ in await kb.lexical.search(q, 10)}
    assert "payroll.md" in {d.metadata["source"] for d, _ in await kb.lexical.search(q, 10, frozenset({"finance"}))}


async def test_any_one_of_several_groups_is_enough(kb):
    for g in ("hr", "finance"):
        assert "payroll.md" in sources(await kb.search("payroll calendar", groups={g}))


async def test_public_documents_stay_visible_to_group_members(kb):
    assert "vpn.md" in sources(await kb.search("GlobalProtect VPN connect", groups={"it-staff"}))


async def test_front_matter_is_not_indexed(kb):
    chunks = kb.vs.get(include=["documents", "metadatas"])
    assert not any("access:" in text for text in chunks["documents"])
    by_source = {m["source"]: m for m in chunks["metadatas"]}
    assert by_source["payroll.md"]["access_groups"] == "finance,hr" and by_source["payroll.md"]["acl_hr"] is True


async def test_restricted_chunks_do_not_crowd_out_permitted_ones(tmp_path):
    """Pre-filtering: 15 restricted chunks match better than the public one, fetch_k is 5. A
    post-filter would return nothing; filtering inside the query still finds the public chunk."""
    docs = tmp_path / "docs"
    docs.mkdir()
    for i in range(15):
        (docs / f"secret{i}.md").write_text(f"---\naccess: security\n---\n# Laptop policy {i}\nlaptop refresh laptop policy {i}\n")
    (docs / "public.md").write_text("# Laptops\nThe laptop refresh policy is every 4 years.\n")
    settings = Settings(offline=True, chroma_dir=tmp_path / "chroma", docs_dir=docs, fetch_k=5, top_k=2, reranker="none")
    kb = KnowledgeBase(settings, HashingEmbeddings())
    kb.ingest()
    assert sources(await kb.search("laptop refresh policy")) == {"public.md"}


async def test_chunks_indexed_before_access_control_are_hidden(kb):
    kb.vs.add_documents([Document(page_content="Old wiki: break-glass approvals go to the duty manager.",
                                  metadata={"source": "old.md", "chunk_id": "old#0", "title": "", "section": ""},
                                  id="old#0")], ids=["old#0"])
    kb.lexical.upsert([Document(page_content="Old wiki: break-glass approvals go to the duty manager.",
                                metadata={"source": "old.md", "chunk_id": "old#0"}, id="old#0")])
    for groups in (set(), {"it-staff"}):
        assert "old.md" not in sources(await kb.search("break-glass approvals duty manager", groups=groups))
    assert kb.unlabelled_sample() == 1


async def test_a_store_that_ignores_the_filter_is_caught(tmp_path, caplog):
    """Defence in depth: a lexical index that returns a restricted chunk anyway (bug, stale
    index) never gets it to the user, and the leak is logged as an error."""
    secret = doc("hr#0", "Salary bands for 2026 are confidential.", source="hr.md", groups=("hr",))
    settings = Settings(offline=True, chroma_dir=tmp_path / "chroma", offline_min_relevance=0.99, reranker="none")
    kb = KnowledgeBase(settings, HashingEmbeddings(), lexical=StaticLexical([(secret, 9.0)]))
    with caplog.at_level(logging.ERROR, logger="northwind_rag.store"):
        assert await kb.search("salary bands") == []
    assert "unreadable" in caplog.text and "hr#0" in caplog.text
    assert [d.metadata["chunk_id"] for d in await kb.search("salary bands", groups={"hr"})] == ["hr#0"]


# --------------------------------------------------------------------------- Elasticsearch
async def test_elasticsearch_stores_groups_and_filters_by_them():
    es = ElasticsearchIndex("http://es:9200", "kb", client=FakeES(), async_client=FakeAsyncES())
    es.upsert([doc("pay#0", "Payroll runs on the 25th.", groups=("hr", "finance"))])
    _, operations, _ = next(c for c in es.es.calls if c[0] == "bulk")
    body = next(op for op in operations if "content" in op)
    assert body["access_groups"] == ["finance", "hr"]
    await es.search("payroll", 5, frozenset({"hr"}))
    assert es.aes.last["query"]["bool"]["filter"] == [{"terms": {"access_groups": ["hr", "public"]}}]


def test_existing_elasticsearch_index_gets_the_new_field():
    fake = FakeES()
    fake.indices.created = {"index": "kb"}  # created by an older version, without access_groups
    calls = []
    fake.indices.put_mapping = lambda index, properties: calls.append((index, properties))
    es = ElasticsearchIndex("http://es:9200", "kb", client=fake, async_client=FakeAsyncES())
    es.ensure_index()
    es.ensure_index()
    assert len(calls) == 1 and calls[0][1]["access_groups"] == {"type": "keyword"}


# ------------------------------------------------------------------------------ end to end
QUESTION = "Who approves a break-glass admin session?"


async def post(pipeline, path: str, headers: dict | None = None) -> httpx.Response:
    app.state.pipeline = pipeline
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        return await client.post(path, json={"question": QUESTION}, headers=headers or {})


async def test_api_member_of_the_group_gets_the_answer(pipeline):
    r = await post(pipeline, "/chat", {"X-Jwt-Payload": jwt_payload({"sub": "ana", "groups": ["it-staff"]})})
    body = r.json()
    assert body["status"] == "answered" and "duty manager" in body["answer"]
    assert any(s.startswith("break-glass-access#") for s in body["sources"])


@pytest.mark.parametrize("headers", [{}, {"X-Jwt-Payload": jwt_payload({"sub": "bo", "groups": ["hr"]})},
                                     {"X-Jwt-Payload": "garbage"}], ids=["no-token-groups", "other-group", "malformed"])
async def test_api_others_learn_nothing(pipeline, headers):
    body = (await post(pipeline, "/chat", headers)).json()
    assert body["status"] != "answered" and "duty manager" not in body["answer"]
    assert not any(s.startswith("break-glass-access#") for s in body["sources"])


async def test_api_stream_applies_the_same_filter(pipeline):
    allowed = await post(pipeline, "/chat/stream", {"X-Jwt-Payload": jwt_payload({"groups": ["it-staff"]})})
    denied = await post(pipeline, "/chat/stream")
    assert "duty manager" in allowed.text and "duty manager" not in denied.text
    assert "break-glass-access#" not in denied.text
