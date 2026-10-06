# Operating guide

## Start, stop, and inspect

Start the loopback demo with `python -m evidence.serve --demo` and open port 8102. Stop with Ctrl-C. Persistent data lives under the configured runtime directory. Restarting does not reset edited demo data. Remove or choose a fresh runtime directory only when you deliberately want a fresh synthetic workspace.

For hosted use, configure `API_KEYS_JSON` with strong random keys and server-owned tenant/user/role claims; omit `APP_DEMO`. Terminate TLS, restrict network access and persist application state. Do not publish the built-in demo credentials on an externally reachable service. The included Compose configurations publish to loopback only.

`/health` checks application availability. The three operational services expose authenticated `/metrics` and optional OpenTelemetry export through `OTEL_EXPORTER_OTLP_ENDPOINT`. These metrics do not log raw prompts, API keys, or document bodies. The evaluation and model services expose health and their explicit run/inference results.

## State and index behavior

The metadata database retains immutable document revisions and tombstones. Every search reads current authorized metadata before finding or building a snapshot. The snapshot cache has four entries per process; each index caches up to 128 query embeddings. Cached dense matrices are keyed by content fingerprint and pinned model revision.

Deleting cached index files is safe while the application is stopped; they can be rebuilt. Deleting the metadata database loses document revision and access history. Back it up through SQLite's backup API. Neural model weights use the standard Hugging Face cache and require network access only on first download.

For pgvector, start `docker compose -f compose.pgvector.yaml up -d`, set `PGVECTOR_DSN`, install/pull `all-minilm` in Ollama, then call `/vectors/reindex` as a reviewer. Index each authorized role as needed. Current metadata still governs access even if vector cleanup or refresh is delayed.

## Generation and credentials

Set `ROUTER_URL` and a tenant-scoped router credential via `SERVICE_KEYS_JSON`, or set `SERVICE_TENANT=alpha` and `ROUTER_API_KEY` for one tenant. A missing tenant mapping rejects generation with 503; search continues to work. Citation validation checks IDs and JSON shape, not semantic entailment. Keep generated prose subject to review.

## Useful failure demonstrations

- Search a distinctive term, revoke analyst access, and repeat the query: no passage returns.
- Revise a document: the next hit carries the new revision and text hash.
- Delete a document: neither search nor its history endpoint exposes it to the previous reader.
- Submit feedback for another user's query ID: expect 404.
- Stop Ollama/pgvector: optional transformer retrieval becomes unavailable; the lexical browser remains usable.


## Five-minute technical walkthrough

Explain the central data flow in the README, run one successful user workflow, deliberately trigger one failure above, inspect its recorded evidence, and explain one limitation you would address before a broader deployment. Describe measured results as results of this repository's experiments rather than as prior employer production outcomes.
