import os
import pytest
from fastapi.testclient import TestClient
from evidence.api import create_app
from evidence.auth import DEMO_KEYS
from evidence.demo import DOCUMENTS
from evidence.pgvector import PgVectorIndex
from evidence.retrieval import answer, graph_path, search
from evidence.store import Store, Document


@pytest.fixture
def store(tmp_path):
    result = Store(tmp_path / "evidence.db")
    for item in DOCUMENTS:
        result.ingest("alpha", Document(**item))
    return result


@pytest.mark.parametrize("method", ["bm25", "latent", "hybrid"])
def test_acl_before_retrieval(store, method):
    visible = store.visible("alpha", "analyst")
    hits = search(visible, "SYNTHETIC ORCHID override secrets", method)
    assert all(h["document_id"] != "restricted" for h in hits)
    assert search(store.visible("beta", "analyst"), "settlement", method) == []


def test_versioning_revocation_and_deletion(store):
    doc = Document(**DOCUMENTS[0])
    assert store.ingest("alpha", doc) == 1
    doc.roles = ["reviewer"]
    assert store.ingest("alpha", doc) == 2
    assert "settlement" not in [d["id"] for d in store.visible("alpha", "analyst")]
    assert store.delete("alpha", doc.id)
    assert not store.delete("alpha", doc.id)
    assert "settlement" not in [d["id"] for d in store.visible("alpha", "reviewer")]


def test_graph_provenance_and_access(store):
    path = graph_path(store.visible("alpha", "analyst"), "settlement-mismatch", "operations")
    assert len(path) == 2
    assert [e["document_id"] for e in path] == ["settlement", "ownership"]
    assert graph_path(store.visible("beta", "analyst"), "settlement-mismatch", "operations") is None


def test_exact_citations_and_abstention(store):
    visible = store.visible("alpha", "analyst")
    result = answer(visible, "settlement date mismatch", "bm25")
    for citation in result["citations"]:
        original = next(doc for doc in visible if doc["id"] == citation["document_id"])
        assert citation["text"] in original["text"]
    assert answer(visible, "xyzunseenword")["abstained"]
    assert answer(visible, "settlement", max_context_words=1)["abstained"]


def test_api_authorization(tmp_path):
    with TestClient(create_app(tmp_path / "api.db", DEMO_KEYS)) as client:
        analyst = {"Authorization": "Bearer demo-analyst"}
        reviewer = {"Authorization": "Bearer demo-reviewer"}
        assert client.post("/documents", headers=analyst, json=DOCUMENTS[0]).status_code == 403
        assert (
            client.post("/documents", headers=reviewer, json=DOCUMENTS[0]).json()["revision"] == 1
        )
        result = client.post("/answer", headers=analyst, json={"question": "settlement"})
        assert not result.json()["abstained"]
        assert client.post("/answer", json={"question": "settlement"}).status_code == 401


@pytest.mark.skipif(
    not os.getenv("PGVECTOR_TEST_DSN"),
    reason="Set PGVECTOR_TEST_DSN for live PostgreSQL integration",
)
def test_pgvector_roundtrip_isolation_and_revision():
    index = PgVectorIndex(os.environ["PGVECTOR_TEST_DSN"])
    index.setup()
    embedding = [1.0] + [0.0] * 383
    for tenant in ("test-alpha", "test-beta"):
        index.upsert(tenant, "one", 2, ["reviewer"], "current", embedding)
    index.upsert("test-alpha", "one", 1, ["analyst"], "stale", embedding)
    assert index.search("test-alpha", "analyst", embedding) == []
    results = index.search("test-alpha", "reviewer", embedding)
    assert len(results) == 1 and results[0]["text"] == "current"
    assert index.search("test-alpha", "reviewer", embedding, allowed_ids=[]) == []
    assert index.search("test-alpha", "reviewer", embedding, allowed_ids=["other"]) == []
    index.delete("test-alpha", "one")
    assert index.search("test-alpha", "reviewer", embedding) == []
    assert len(index.search("test-beta", "reviewer", embedding)) == 1
    index.delete("test-beta", "one")
