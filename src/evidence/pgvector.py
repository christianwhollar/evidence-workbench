"""Optional persistent dense index for externally generated, fixed-size embeddings."""

import math
import psycopg


def vector_literal(values, dimensions=384):
    if len(values) != dimensions or not all(math.isfinite(v) for v in values):
        raise ValueError(f"Expected {dimensions} finite embedding values")
    if not any(values):
        raise ValueError("Zero vectors have undefined cosine similarity")
    return "[" + ",".join(str(float(v)) for v in values) + "]"


class PgVectorIndex:
    def __init__(self, dsn):
        self.dsn = dsn

    def setup(self):
        with psycopg.connect(self.dsn) as db:
            db.execute("CREATE EXTENSION IF NOT EXISTS vector")
            db.execute("""CREATE TABLE IF NOT EXISTS evidence_vectors (
                tenant text NOT NULL, id text NOT NULL, revision integer NOT NULL,
                roles text[] NOT NULL, text text NOT NULL, embedding vector(384) NOT NULL,
                PRIMARY KEY(tenant,id))""")

    def upsert(self, tenant, document_id, revision, roles, text, embedding):
        with psycopg.connect(self.dsn) as db:
            db.execute(
                """INSERT INTO evidence_vectors VALUES(%s,%s,%s,%s,%s,%s::vector)
                ON CONFLICT(tenant,id) DO UPDATE SET revision=EXCLUDED.revision,
                roles=EXCLUDED.roles,text=EXCLUDED.text,embedding=EXCLUDED.embedding
                WHERE evidence_vectors.revision<=EXCLUDED.revision""",
                (tenant, document_id, revision, roles, text, vector_literal(embedding)),
            )

    def search(self, tenant, role, embedding, limit=5, allowed_ids=None):
        if not 1 <= limit <= 50:
            raise ValueError("limit must be 1..50")
        if allowed_ids == []:
            return []
        authorization = " AND id=ANY(%s)" if allowed_ids is not None else ""
        parameters = [vector_literal(embedding), tenant, role]
        if allowed_ids is not None:
            parameters.append(allowed_ids)
        parameters.extend([vector_literal(embedding), limit])
        with psycopg.connect(self.dsn) as db:
            rows = db.execute(
                """SELECT id,revision,text,1-(embedding <=> %s::vector) score
                FROM evidence_vectors WHERE tenant=%s AND %s=ANY(roles)
                """
                + authorization
                + " ORDER BY embedding <=> %s::vector,id LIMIT %s",
                parameters,
            ).fetchall()
        return [dict(zip(("id", "revision", "text", "score"), row)) for row in rows]

    def delete(self, tenant, document_id):
        with psycopg.connect(self.dsn) as db:
            db.execute(
                "DELETE FROM evidence_vectors WHERE tenant=%s AND id=%s", (tenant, document_id)
            )
