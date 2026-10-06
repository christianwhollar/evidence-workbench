# Retrieval, versions, and failure modes

Authorization comes from the server's API-key mapping. Request bodies cannot choose a tenant or role. Ingestion creates an immutable revision when text, roles, or edges change. Repeating the same document is idempotent. Deletion adds a tombstone. Current visibility is calculated from the newest revision before any retrieval or graph traversal.

The default pipeline splits on words with overlap, ranks BM25 and TF-IDF/SVD independently, and fuses ranks with reciprocal rank fusion. It fits SVD only on visible content; this avoids leaking hidden corpus statistics but is intentionally inefficient for a large corpus. Precomputed tenant-specific embeddings and asynchronous indexing would be the next step.

The pgvector adapter stores fixed-size, externally generated embeddings and uses exact cosine search. There is no ANN index at this corpus size. Dense searches carry current chunk IDs from the governed source store into the SQL WHERE clause. Old revisions and newly restricted content are excluded even if their vectors remain in the database. Cleanup of obsolete vectors is an operational follow-up; correctness does not depend on cleanup.

Reindexing uses the reviewer's authorized documents. A document with only an analyst role requires a separately authorized indexing job; the service does not silently expand the reviewer's read permissions. A failed partial indexing run can temporarily reduce recall. It does not grant access to hidden passages.

Exact excerpts are the safest baseline here. A lexical match does not prove that a passage answers the question. Transformer answers use a demo cosine cutoff of 0.25; it is not calibrated confidence. Retrieved instructions are treated as untrusted, and generation cannot call tools. Citation validation prevents invented source IDs but cannot prevent unsupported claims referencing valid IDs. The caller must evaluate answer correctness separately.

The graph is a bounded directed BFS over explicit triples, not a knowledge extraction model. Every hop carries source document and revision. This makes structured reasoning inspectable.

## Walkthrough

Show a query, its exact source excerpt, and its revision. Change a document's roles without refreshing pgvector and prove it disappears. Compare a paraphrase across BM25 and the local transformer. Explain why an unrelated query can still have a nearest vector. Finally, trace the two-hop team ownership path.

References: [pgvector](https://github.com/pgvector/pgvector), [Ollama embeddings API](https://docs.ollama.com/api/embed).
