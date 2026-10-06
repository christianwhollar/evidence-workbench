"""Real local transformer embeddings backed by the pgvector database."""

import os
import httpx
from .pgvector import PgVectorIndex
from .retrieval import chunks


class OllamaEmbeddings:
    def __init__(self, url=None, model=None):
        self.url = url or os.getenv("EMBEDDING_URL", "http://127.0.0.1:11434")
        self.model = model or os.getenv("EMBEDDING_MODEL", "all-minilm")

    def encode(self, texts):
        response = httpx.post(
            self.url.rstrip("/") + "/api/embed",
            json={"model": self.model, "input": texts, "truncate": False},
            timeout=30,
        )
        response.raise_for_status()
        embeddings = response.json()["embeddings"]
        if len(embeddings) != len(texts) or any(len(v) != 384 for v in embeddings):
            raise ValueError("Expected one 384-dimensional embedding per input")
        return embeddings


def index_visible(store, tenant, role, dsn):
    """Index only documents this operator can read. Reindex under each authorized role as needed."""
    documents = store.visible(tenant, role)
    pieces = chunks(documents)
    if not pieces:
        return 0
    vectors = OllamaEmbeddings().encode([p["text"] for p in pieces])
    index = PgVectorIndex(dsn)
    index.setup()
    by_id = {doc["id"]: doc for doc in documents}
    for piece, vector in zip(pieces, vectors):
        index.upsert(
            tenant,
            piece["id"],
            piece["revision"],
            by_id[piece["document_id"]]["roles"],
            piece["text"],
            vector,
        )
    return len(pieces)


def search_dense(store, tenant, role, query, dsn, limit=5):
    # Current SQLite ACLs and versions form the allowlist BEFORE the SQL vector search.
    # A stale vector index cannot undo a revocation, deletion, or revision update.
    pieces = chunks(store.visible(tenant, role))
    metadata = {piece["id"]: piece for piece in pieces}
    if not metadata:
        return []
    vector = OllamaEmbeddings().encode([query])[0]
    hits = PgVectorIndex(dsn).search(tenant, role, vector, limit=limit, allowed_ids=list(metadata))
    return [{**metadata[hit["id"]], "score": float(hit["score"])} for hit in hits]
