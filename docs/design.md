# Architecture and decisions

The README describes the current architecture and measured results. This document records the deployment boundary and links decisions to inspectable code.

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


## Evolution

The original compact reference implementation is documented in [design-v1.md](design-v1.md). Version 0.2 adds a complete browser workflow, operational state handling, larger experiments or failure studies, and reproducible release artifacts. Earlier studies remain in `reports/`; they have not been replaced with improved numbers under their original names.

See [operations.md](operations.md) for setup and failure demonstrations and [verification.md](verification.md) for the exact validation scope.

## Interface design

Warm paper tones, serif headings, and a reading column with source metadata in the margin. This research library uses its own typography, spacing, navigation and component shapes. Fonts and icons are local system fonts and inline SVG, with no external asset requests. The interface supports narrow screens, visible keyboard focus, a skip link, and active navigation semantics.
