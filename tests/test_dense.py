import os
import pytest
from evidence.demo import DOCUMENTS
from evidence.dense import index_visible, search_dense
from evidence.store import Store, Document


@pytest.mark.skipif(
    not (os.getenv("PGVECTOR_TEST_DSN") and os.getenv("RUN_EMBEDDING_TESTS") == "1"),
    reason="Needs pgvector and local all-minilm model",
)
def test_live_transformer_and_stale_index_revocation(tmp_path):
    store = Store(tmp_path / "dense.db")
    for item in DOCUMENTS:
        store.ingest("dense-test", Document(**item))
    dsn = os.environ["PGVECTOR_TEST_DSN"]
    assert index_visible(store, "dense-test", "reviewer", dsn) == 4
    hits = search_dense(store, "dense-test", "analyst", "dates differ settlement", dsn)
    assert hits and all(hit["document_id"] != "restricted" for hit in hits)
    store.ingest("dense-test", Document(**{**DOCUMENTS[0], "roles": ["reviewer"]}))
    hits = search_dense(store, "dense-test", "analyst", "dates differ settlement", dsn)
    assert all(hit["document_id"] not in {"restricted", "settlement"} for hit in hits)
