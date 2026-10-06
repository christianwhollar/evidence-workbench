# Evidence workbench

[![checks](https://github.com/christianwhollar/evidence-workbench/actions/workflows/ci.yml/badge.svg)](https://github.com/christianwhollar/evidence-workbench/actions/workflows/ci.yml)

Search governed documents with BM25, latent semantic retrieval, reciprocal-rank fusion, or local transformer embeddings in PostgreSQL/pgvector. Every returned passage retains its document revision and source hash.

This is a runnable reference implementation using synthetic data. See [design notes](docs/design.md), [development provenance](DEVELOPMENT.md), and [verification](docs/verification.md).

## Run locally

Python 3.12 is the tested runtime.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -c constraints.txt -e '.[dev]'
python -m evidence.demo
pytest -q
```

## Retrieval and governance

```mermaid
flowchart LR
    Ingest[Versioned ingestion] --> Docs[(SQLite documents and roles)]
    Docs --> ACL[Current tenant and role filter]
    ACL --> Local[BM25 / latent / rank fusion]
    ACL --> Allowlist[Current chunk ID allowlist]
    Allowlist --> Vector[(pgvector cosine search)]
    Local --> Context[Budgeted source context]
    Vector --> Context
    Context --> Excerpts[Exact cited excerpts]
    Context --> LLM[Optional generated answer]
```

The no-download default fits a small TF-IDF/SVD latent space over authorized documents only. This is a classical semantic baseline, not a pretrained embedding model. The optional transformer route uses Ollama's `all-minilm` model and 384-dimensional pgvector embeddings. The graph endpoint follows explicitly supplied relationships and returns document provenance for each hop.

## Start and seed the service

```bash
APP_DEMO=1 uvicorn evidence.api:app --host 127.0.0.1 --port 8102
# In a second terminal:
python scripts/seed_service.py
```

Use `http://127.0.0.1:8102/docs` or:

```bash
curl http://127.0.0.1:8102/answer \
  -H 'Authorization: Bearer demo-analyst' -H 'Content-Type: application/json' \
  -d '{"question":"What happens when settlement dates differ?","method":"hybrid"}'
```

Reviewer credentials (`demo-reviewer`) can ingest, delete, and index documents. Analyst credentials see only analyst-visible documents. Tenant beta (`demo-beta`) cannot read alpha's data. Demo keys are enabled only when `APP_DEMO=1`; normal mode requires server-configured `API_KEYS_JSON`.

## Real transformer retrieval

```bash
docker compose -f compose.pgvector.yaml up -d --wait
ollama pull all-minilm
export PGVECTOR_DSN='postgresql://postgres:local-demo-only@127.0.0.1:55432/evidence'
APP_DEMO=1 uvicorn evidence.api:app --host 127.0.0.1 --port 8102
# Seed first, then index:
python scripts/seed_service.py
curl -X POST http://127.0.0.1:8102/vectors/reindex -H 'Authorization: Bearer demo-reviewer'
```

Use `"method":"transformer"` in `/search` or `/answer`. Stop any prior service before starting the configured one. In containers, set `EMBEDDING_URL` to a reachable Ollama address. The SQLite ACL/version allowlist is applied inside the vector SQL query, so a stale index cannot undo a revocation. Embedding changes require a new index; the default schema expects 384 dimensions.

```bash
PGVECTOR_TEST_DSN="$PGVECTOR_DSN" RUN_EMBEDDING_TESTS=1 pytest -q
```

## Generated answers and graphs

Set `ROUTER_URL` and `ROUTER_API_KEY` to use [inference-router](https://github.com/christianwhollar/inference-router), then send `"generate":true` to `/answer`. Evidence is delimited as untrusted content. Unknown citation IDs or invalid JSON trigger fallback to excerpts. Citation membership is checked; entailment is not. The response makes that distinction explicit.

`POST /graph/path` accepts `source`, `target`, and bounded `max_hops`. Try `settlement-mismatch` to `operations` after seeding. All graph edges originate from accessible document versions.

## Evidence

[Integration output](reports/integration.json) records a real local generated answer, transformer retrieval, and access revocation without vector reindexing. [Agent eval lab](https://github.com/christianwhollar/agent-eval-lab) contains the paired retrieval comparison. The corpus is deliberately small; it tests the pipeline and cannot rank retrieval methods in general.
