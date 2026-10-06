"""Evaluate every SciFact test query against the complete BEIR corpus."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import time
import zipfile

import httpx
import numpy as np

from .index import EMBEDDER, EMBEDDER_REVISION, RERANKER, RERANKER_REVISION, Index, Models

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
SHA256 = "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"


def load(cache):
    path = Path(cache) / "scifact.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        response = httpx.get(URL, follow_redirects=True, timeout=90)
        response.raise_for_status()
        path.write_bytes(response.content)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError("SciFact archive checksum mismatch")
    with zipfile.ZipFile(path) as archive:
        corpus = [json.loads(line) for line in archive.read("scifact/corpus.jsonl").splitlines()]
        queries = {
            r["_id"]: r["text"]
            for r in map(json.loads, archive.read("scifact/queries.jsonl").splitlines())
        }
        raw = archive.read("scifact/qrels/test.tsv").decode()
    qrels = {}
    for row in csv.DictReader(io.StringIO(raw), delimiter="\t"):
        if int(row["score"]) > 0:
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    pieces = [
        {
            "id": r["_id"],
            "document_id": r["_id"],
            "title": r["title"],
            "text": r["text"],
            "revision": 1,
        }
        for r in corpus
    ]
    return pieces, queries, qrels


def retrieval_metrics(ranked, relevant, k=10):
    if not relevant:
        raise ValueError("Benchmark metrics require at least one relevant document")
    top = list(dict.fromkeys(ranked))[:k]
    gains = [2 ** relevant.get(doc, 0) - 1 for doc in top]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(
        (2**value - 1) / math.log2(i + 2)
        for i, value in enumerate(sorted(relevant.values(), reverse=True)[:k])
    )
    ranks = [i + 1 for i, doc in enumerate(top) if doc in relevant]
    return {
        "ndcg_at_10": dcg / ideal,
        "recall_at_10": len(ranks) / len(relevant),
        "mrr_at_10": 1 / ranks[0] if ranks else 0,
    }


def bootstrap(values, seed=17):
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (2000, len(values))).mean(1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def run(output, cache="runtime/benchmark", device="cpu"):
    import torch

    torch.set_num_threads(4)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pieces, queries, qrels = load(cache)
    models = Models(device)
    index = Index(pieces, models, Path(cache) / "embeddings")
    before = time.perf_counter()
    index.dense_vectors()
    build_seconds = time.perf_counter() - before
    # Separate model loading/warmup from steady-state query timings.
    models.rerank("warmup", [pieces[0]["text"]])
    config = {
        "corpus": "BEIR SciFact",
        "source": URL,
        "archive_sha256": SHA256,
        "corpus_documents": len(pieces),
        "test_queries": len(qrels),
        "embedder": EMBEDDER,
        "embedder_revision": EMBEDDER_REVISION,
        "reranker": RERANKER,
        "reranker_revision": RERANKER_REVISION,
        "dense_input": "title + abstract, model's 256 wordpiece truncation",
        "reranker_max_length": 512,
        "candidates": 50,
        "rrf_constant": 60,
        "bm25": {"k1": 1.5, "b": 0.75},
        "k": 10,
        "device": device,
        "platform": platform.platform(),
        "torch_threads": 4,
        "model_loading_and_index_build_excluded_from_query_latency": True,
        "embedding_build_seconds": build_seconds,
        "scope": "Off-the-shelf models; no SciFact label fitting or parameter tuning.",
    }
    (output / "protocol.json").write_text(json.dumps(config, indent=2))
    rows = []
    with (output / "queries.jsonl").open("w") as stream:
        for number, query_id in enumerate(sorted(qrels)):
            row = {"query_id": query_id, "relevant": qrels[query_id], "methods": {}}
            for method in ["bm25", "dense", "fusion", "rerank"]:
                index.query_vectors.clear()
                started = time.perf_counter()
                hits = index.rank(queries[query_id], method, limit=10)
                elapsed = (time.perf_counter() - started) * 1000
                ids = [h["id"] for h in hits]
                row["methods"][method] = {
                    **retrieval_metrics(ids, qrels[query_id]),
                    "latency_ms": elapsed,
                    "retrieved": ids,
                }
            rows.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if number % 25 == 0:
                print(f"Evaluated {number + 1}/{len(qrels)} queries", flush=True)
    summary = {}
    for method in ["bm25", "dense", "fusion", "rerank"]:
        values = [r["methods"][method] for r in rows]
        summary[method] = {
            name: float(np.mean([v[name] for v in values]))
            for name in ["ndcg_at_10", "recall_at_10", "mrr_at_10"]
        }
        summary[method].update(
            {
                "ndcg_ci95": bootstrap([v["ndcg_at_10"] for v in values]),
                "p50_ms": float(np.median([v["latency_ms"] for v in values])),
                "p95_ms": float(np.quantile([v["latency_ms"] for v in values], 0.95)),
            }
        )
    delta = [
        r["methods"]["rerank"]["ndcg_at_10"] - r["methods"]["bm25"]["ndcg_at_10"] for r in rows
    ]
    report = {
        "config": config,
        "summary": summary,
        "rerank_minus_bm25": {
            "ndcg_difference": float(np.mean(delta)),
            "paired_ci95": bootstrap(delta),
        },
        "limitations": [
            "Scientific retrieval does not validate financial-domain accuracy.",
            "Qrels measure relevance, not claim entailment or answer faithfulness.",
            "Wall-clock timings depend on device load and are not a service SLA.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", default="runtime/benchmark")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args.output, args.cache, args.device)


if __name__ == "__main__":
    main()
