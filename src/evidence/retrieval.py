"""A transparent BM25/latent-semantic baseline with rank fusion and provenance."""

import hashlib
import math
import re
from collections import Counter, deque
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def chunks(documents, words=90, overlap=15):
    if not 0 <= overlap < words:
        raise ValueError("Require 0 <= overlap < words")
    result = []
    for doc in documents:
        spans = list(re.finditer(r"\S+", doc["text"]))
        for start in range(0, len(spans), words - overlap):
            end = min(start + words, len(spans))
            text = doc["text"][spans[start].start() : spans[end - 1].end()]
            chunk_id = f"{doc['id']}@{doc['revision']}:{start}"
            result.append(
                {
                    "id": chunk_id,
                    "document_id": doc["id"],
                    "revision": doc["revision"],
                    "title": doc["title"],
                    "text": text,
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
            if end == len(spans):
                break
    return result


def bm25(texts, query):
    counts = [Counter(tokens(text)) for text in texts]
    average = sum(map(lambda x: sum(x.values()), counts)) / max(len(counts), 1)
    scores = np.zeros(len(texts))
    for term in set(tokens(query)):
        frequency = sum(term in count for count in counts)
        idf = math.log(1 + (len(texts) - frequency + 0.5) / (frequency + 0.5))
        for i, count in enumerate(counts):
            tf = count[term]
            scores[i] += (
                idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * sum(count.values()) / max(average, 1)))
            )
    return scores


def latent_vectors(texts, query):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    q = vectorizer.transform([query])
    dimensions = min(32, matrix.shape[0] - 1, matrix.shape[1] - 1)
    if dimensions >= 2:
        svd = TruncatedSVD(n_components=dimensions, random_state=17)
        docs, query_vector = svd.fit_transform(matrix), svd.transform(q)
    else:
        docs, query_vector = matrix.toarray(), q.toarray()
    return normalize(docs), normalize(query_vector)[0]


def search(documents, query, method="hybrid", limit=5):
    if method not in {"bm25", "latent", "hybrid"}:
        raise ValueError("Unknown retrieval method")
    pieces = chunks(documents)
    if not pieces or not tokens(query):
        return []
    texts = [chunk["text"] for chunk in pieces]
    lexical = bm25(texts, query)
    # Fit only authorized documents. Hidden text cannot influence the latent space.
    try:
        vectors, q = latent_vectors(texts, query)
        semantic = vectors @ q
    except ValueError:
        semantic = np.zeros(len(pieces))
    if method == "bm25":
        scores = lexical
    elif method == "latent":
        scores = semantic
    else:
        scores = np.zeros(len(pieces))
        for source in (lexical, semantic):
            for rank, index in enumerate(np.argsort(-source, kind="stable")):
                if source[index] > 1e-8:
                    scores[index] += 1 / (60 + rank + 1)
    order = np.argsort(-scores, kind="stable")
    return [
        {**pieces[i], "score": float(scores[i]), "lexical_score": float(lexical[i])}
        for i in order[:limit]
        if scores[i] > 1e-8
    ]


def graph_path(documents, source, target, max_hops=3):
    if not 1 <= max_hops <= 5:
        raise ValueError("max_hops must be between 1 and 5")
    adjacency = {}
    for doc in documents:
        for edge in doc["edges"]:
            adjacency.setdefault(edge["source"], []).append(
                {**edge, "document_id": doc["id"], "revision": doc["revision"]}
            )
    queue = deque([(source, [])])
    seen = {source}
    while queue:
        node, path = queue.popleft()
        if node == target:
            return path
        if len(path) >= max_hops:
            continue
        for edge in adjacency.get(node, []):
            if edge["target"] not in seen:
                seen.add(edge["target"])
                queue.append((edge["target"], path + [edge]))
    return None


def answer(documents, query, method="hybrid", max_context_words=240):
    hits = search(documents, query, method)
    return answer_from_hits(hits, max_context_words)


def answer_from_hits(hits, max_context_words=240):
    selected, used = [], 0
    for hit in hits:
        count = len(hit["text"].split())
        if used + count <= max_context_words:
            selected.append(hit)
            used += count
    if not selected:
        return {
            "abstained": True,
            "answer": "No accessible evidence matched the question.",
            "citations": [],
            "mode": "extractive",
        }
    # Exact excerpts are the baseline. A matching passage is not proof it answers the question.
    return {
        "abstained": False,
        "answer": "\n\n".join(f"[{c['id']}] {c['text']}" for c in selected),
        "citations": selected,
        "mode": "extractive",
        "context_words": used,
    }
