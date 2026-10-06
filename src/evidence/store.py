import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field


class Edge(BaseModel):
    source: str = Field(min_length=1, max_length=120)
    relation: str = Field(min_length=1, max_length=120)
    target: str = Field(min_length=1, max_length=120)


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=100000)
    roles: list[str] = Field(min_length=1, max_length=10)
    edges: list[Edge] = Field(default_factory=list, max_length=100)


class Store:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS documents (
                    tenant TEXT NOT NULL, id TEXT NOT NULL, revision INTEGER NOT NULL,
                    fingerprint TEXT NOT NULL, payload TEXT NOT NULL, deleted INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(tenant,id,revision)
                );
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def ingest(self, tenant, document):
        payload = document.model_dump(mode="json")
        payload["roles"] = sorted(set(payload["roles"]))
        encoded = json.dumps(payload, sort_keys=True)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute(
                "SELECT * FROM documents WHERE tenant=? AND id=? ORDER BY revision DESC LIMIT 1",
                (tenant, document.id),
            ).fetchone()
            if old and old["fingerprint"] == digest and not old["deleted"]:
                return old["revision"]
            revision = old["revision"] + 1 if old else 1
            db.execute(
                "INSERT INTO documents VALUES(?,?,?,?,?,0)",
                (tenant, document.id, revision, digest, encoded),
            )
        return revision

    def delete(self, tenant, document_id):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute(
                "SELECT * FROM documents WHERE tenant=? AND id=? ORDER BY revision DESC LIMIT 1",
                (tenant, document_id),
            ).fetchone()
            if not old or old["deleted"]:
                return False
            db.execute(
                "INSERT INTO documents VALUES(?,?,?,?,?,1)",
                (tenant, document_id, old["revision"] + 1, old["fingerprint"], old["payload"]),
            )
        return True

    def visible(self, tenant, role):
        with self.connect() as db:
            rows = db.execute(
                """SELECT d.* FROM documents d JOIN
                (SELECT id,MAX(revision) revision FROM documents WHERE tenant=? GROUP BY id) latest
                ON d.id=latest.id AND d.revision=latest.revision WHERE d.tenant=? AND d.deleted=0
                ORDER BY d.id""",
                (tenant, tenant),
            ).fetchall()
        docs = []
        for row in rows:
            payload = json.loads(row["payload"])
            if role in payload["roles"]:
                docs.append({**payload, "revision": row["revision"]})
        return docs
