"""Reusable lexical/dense snapshots, reciprocal rank fusion and cross-encoder reranking.

Build snapshots from authorized documents only. The snapshot fingerprint includes every
revision and role, so permission changes cannot reuse a previously visible corpus.
"""

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import threading

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
RERANKER = "cross-encoder/ms-marco-TinyBERT-L2-v2"
RERANKER_REVISION = "81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc"


class Models:
    def __init__(self, device="cpu"):
        self.device, self.encoder, self.reranker = device, None, None
        self.lock = threading.RLock()

    def encode(self, texts):
        with self.lock:
            if self.encoder is None:
                from sentence_transformers import SentenceTransformer

                self.encoder = SentenceTransformer(
                    EMBEDDER, revision=EMBEDDER_REVISION, device=self.device
                )
            return self.encoder.encode(
                texts,
                normalize_embeddings=True,
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

    def rerank(self, query, texts):
        with self.lock:
            if self.reranker is None:
                from sentence_transformers import CrossEncoder

                self.reranker = CrossEncoder(
                    RERANKER, revision=RERANKER_REVISION, device=self.device, max_length=512
                )
            return np.asarray(
                self.reranker.predict(
                    [(query, text) for text in texts], batch_size=32, show_progress_bar=False
                )
            ).reshape(-1)


class Index:
    def __init__(self, pieces, models=None, cache=None):
        if not pieces:
            raise ValueError("Cannot index an empty corpus")
        self.pieces, self.models = pieces, models or Models()
        self.texts = [p.get("title", "") + "\n" + p["text"] for p in pieces]
        self.fingerprint = hashlib.sha256(json.dumps(pieces, sort_keys=True).encode()).hexdigest()
        self.cache = Path(cache) if cache else None
        self.vectors = None
        self.vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", dtype=np.float64)
        counts = self.vectorizer.fit_transform(self.texts).tocsr()
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        df = np.asarray((counts > 0).sum(axis=0)).ravel()
        idf = np.log(1 + (len(pieces) - df + 0.5) / (df + 0.5))
        row_ids = np.repeat(np.arange(len(pieces)), np.diff(counts.indptr))
        norm = 1.5 * (0.25 + 0.75 * lengths / max(lengths.mean(), 1))
        counts.data = counts.data * 2.5 / (counts.data + norm[row_ids]) * idf[counts.indices]
        self.bm25_matrix = counts
        self.query_vectors = OrderedDict()
        self.lock = threading.RLock()

    def dense_vectors(self):
        with self.lock:
            if self.vectors is None:
                name = hashlib.sha256((self.fingerprint + EMBEDDER_REVISION).encode()).hexdigest()
                path = self.cache / (name + ".npy") if self.cache else None
                if path and path.exists():
                    values = np.load(path, allow_pickle=False)
                else:
                    values = self.models.encode(self.texts)
                    if path:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        temporary = path.with_suffix(".tmp")
                        with temporary.open("wb") as stream:
                            np.save(stream, values, allow_pickle=False)
                        temporary.replace(path)
                if values.shape != (len(self.pieces), 384) or not np.isfinite(values).all():
                    raise ValueError("Embedding cache does not match snapshot")
                self.vectors = values
            return self.vectors

    def query_vector(self, query):
        with self.lock:
            if query not in self.query_vectors:
                self.query_vectors[query] = self.models.encode([query])[0]
            self.query_vectors.move_to_end(query)
            while len(self.query_vectors) > 128:
                self.query_vectors.popitem(last=False)
            return self.query_vectors[query]

    def rank(self, query, method="bm25", limit=10, candidates=50):
        if method not in {"bm25", "dense", "fusion", "rerank"}:
            raise ValueError("Unknown retrieval method")
        if not 1 <= limit <= 100 or not limit <= candidates <= 200:
            raise ValueError("Require 1 <= limit <= candidates <= 200")
        q = self.vectorizer.transform([query])
        q.data[:] = 1
        lexical = (self.bm25_matrix @ q.T).toarray().ravel()
        dense = None
        if method != "bm25":
            dense = self.dense_vectors() @ self.query_vector(query)
        scores = lexical.copy() if method == "bm25" else dense.copy()
        if method in {"fusion", "rerank"}:
            scores = np.zeros(len(self.pieces))
            for source in (lexical, dense):
                for rank, i in enumerate(np.argsort(-source, kind="stable")[:candidates]):
                    if source[i] > 0:
                        scores[i] += 1 / (60 + rank + 1)
        order = [int(i) for i in np.argsort(-scores, kind="stable")[:candidates] if scores[i] > 0]
        reranked = {}
        if method == "rerank" and order:
            values = self.models.rerank(query, [self.texts[i] for i in order])
            reranked = dict(zip(order, map(float, values)))
            order.sort(key=lambda i: (-reranked[i], i))
        return [
            {
                **self.pieces[i],
                "score": reranked.get(i, float(scores[i])),
                "lexical_score": float(lexical[i]),
                "dense_score": float(dense[i]) if dense is not None else None,
                "fusion_score": float(scores[i]) if method in {"fusion", "rerank"} else None,
            }
            for i in order[:limit]
        ]


class IndexPool:
    """A bounded process-local cache; authoritative ACLs are read before every lookup."""

    def __init__(self, models=None, cache=None, capacity=4):
        self.models, self.cache, self.capacity = models or Models(), cache, capacity
        self.indices = OrderedDict()
        self.lock = threading.RLock()

    def get(self, tenant, role, documents):
        from .retrieval import chunks

        key = hashlib.sha256(
            json.dumps([tenant, role, documents], sort_keys=True).encode()
        ).hexdigest()
        with self.lock:
            if key not in self.indices:
                self.indices[key] = Index(
                    chunks(documents, words=120, overlap=20),
                    self.models,
                    Path(self.cache) / key if self.cache else None,
                )
            self.indices.move_to_end(key)
            while len(self.indices) > self.capacity:
                self.indices.popitem(last=False)
            return self.indices[key]
