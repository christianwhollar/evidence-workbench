import numpy as np
from fastapi.testclient import TestClient
from evidence.api import create_app
from evidence.auth import DEMO_KEYS
from evidence.index import Index, IndexPool
from evidence.store import Document, Store
from evidence.benchmark import retrieval_metrics


def doc(text="Settlement discrepancies go to operations.", roles=None):
    return Document(
        id="policy", title="Settlement policy", text=text, roles=roles or ["analyst", "reviewer"]
    )


def test_index_snapshot_revocation_and_tenant_isolation(tmp_path):
    store = Store(tmp_path / "documents.db")
    store.ingest("alpha", doc())
    pool = IndexPool()
    visible = store.visible("alpha", "analyst")
    first = pool.get("alpha", "analyst", visible)
    assert first.rank("settlement")[0]["document_id"] == "policy"
    assert first is pool.get("alpha", "analyst", visible)
    store.ingest("alpha", doc("Cash differences go to risk."))
    second = pool.get("alpha", "analyst", store.visible("alpha", "analyst"))
    assert second is not first
    assert second.rank("settlement")  # title is indexed, too
    store.ingest("alpha", doc(roles=["reviewer"]))
    assert store.visible("alpha", "analyst") == []
    assert store.history("alpha", "analyst", "policy") == []
    assert store.visible("beta", "reviewer") == []


def test_document_edit_search_feedback_and_revocation(tmp_path):
    a = {"Authorization": "Bearer demo-analyst"}
    r = {"Authorization": "Bearer demo-reviewer"}
    with TestClient(create_app(tmp_path / "api.db", DEMO_KEYS)) as c:
        assert c.post("/documents", headers=r, json=doc().model_dump()).status_code == 200
        found = c.post(
            "/search", headers=a, json={"question": "settlement", "method": "bm25"}
        ).json()
        assert found["hits"][0]["revision"] == 1
        assert (
            c.post("/feedback/" + found["query_id"], headers=a, json={"useful": True}).status_code
            == 200
        )
        assert (
            c.post("/feedback/" + found["query_id"], headers=r, json={"useful": True}).status_code
            == 404
        )
        assert (
            c.post("/documents", headers=r, json=doc(roles=["reviewer"]).model_dump()).json()[
                "revision"
            ]
            == 2
        )
        assert c.post("/search", headers=a, json={"question": "settlement"}).json()["hits"] == []
        assert c.get("/documents/policy/history", headers=a).status_code == 404
        assert c.delete("/documents/policy", headers=r).json()["deleted"]
        assert not c.get("/documents", headers=r).json()["items"]
        activity = c.get("/activity", headers=a).json()["items"]
        assert all("question" not in item for item in activity)


class FakeModels:
    def encode(self, texts):
        values = np.zeros((len(texts), 384))
        values[:, 0] = 1
        return values

    def rerank(self, query, texts):
        return np.arange(len(texts), dtype=float)


def test_dense_cache_reused_and_reranking_order(tmp_path):
    pieces = [
        {"id": str(i), "document_id": str(i), "title": "Policy", "text": text}
        for i, text in enumerate(["settlement rules", "cash settlement"])
    ]
    index = Index(pieces, FakeModels(), tmp_path)
    values = index.dense_vectors()
    assert index.dense_vectors() is values
    reloaded = Index(pieces, FakeModels(), tmp_path)
    assert np.array_equal(reloaded.dense_vectors(), values)
    hits = index.rank("settlement", "rerank", limit=2)
    assert hits[0]["score"] >= hits[1]["score"]


def test_metrics_use_grades_and_deduplicate():
    values = retrieval_metrics(["a", "a", "b"], {"a": 2, "b": 1})
    assert values["ndcg_at_10"] == 1 and values["recall_at_10"] == 1
    assert retrieval_metrics(["x"], {"a": 1})["mrr_at_10"] == 0


def test_generation_requires_matching_tenant_credential(tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("ROUTER_API_KEY", "alpha-only")
    monkeypatch.setenv("SERVICE_TENANT", "alpha")
    with TestClient(create_app(tmp_path / "scope.db", DEMO_KEYS)) as c:
        response = c.post(
            "/answer",
            headers={"Authorization": "Bearer demo-beta"},
            json={"question": "policy", "generate": True},
        )
        assert response.status_code == 503
        assert "tenant" in response.json()["detail"]
