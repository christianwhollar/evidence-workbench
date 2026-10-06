import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from .auth import configured_keys, identity
from .generation import generate
from .dense import index_visible, search_dense
from .observe import instrument
from .retrieval import answer, answer_from_hits, graph_path, search
from .store import Document, Store
from .index import IndexPool


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=3000)
    method: Literal["bm25", "latent", "hybrid", "transformer", "dense", "fusion", "rerank"] = "bm25"
    generate: bool = False
    limit: int = Field(default=5, ge=1, le=20)


class Feedback(BaseModel):
    useful: bool
    note: str = Field(default="", max_length=1000)


class GraphQuery(BaseModel):
    source: str = Field(min_length=1, max_length=120)
    target: str = Field(min_length=1, max_length=120)
    max_hops: int = Field(default=3, ge=1, le=5)


def create_app(path=None, keys=None):
    @asynccontextmanager
    async def lifespan(app):
        if not app.state.keys:
            raise RuntimeError("Set API_KEYS_JSON or APP_DEMO=1")
        app.state.store = Store(path or Path(os.getenv("DATA_DIR", "runtime")) / "evidence.db")
        app.state.indices = IndexPool(cache=Path(os.getenv("DATA_DIR", "runtime")) / "indices")
        yield
        app.state.tracing.shutdown()

    app = FastAPI(title="Evidence workbench", lifespan=lifespan)
    app.state.keys = configured_keys() if keys is None else keys
    instrument(app, "evidence-workbench")

    @app.get("/health")
    def health():
        with app.state.store.connect() as db:
            db.execute("SELECT 1")
        return {"status": "ok"}

    @app.post("/documents")
    def ingest(document: Document, actor=Depends(identity)):
        if actor.role != "reviewer":
            raise HTTPException(403, "Reviewer required for ingestion")
        return {"id": document.id, "revision": app.state.store.ingest(actor.tenant, document)}

    @app.get("/documents")
    def documents(actor=Depends(identity)):
        return {"items": app.state.store.visible(actor.tenant, actor.role)}

    @app.get("/documents/{document_id}/history")
    def history(document_id: str, actor=Depends(identity)):
        rows = app.state.store.history(actor.tenant, actor.role, document_id)
        if not rows:
            raise HTTPException(404, "Document not found")
        return {"items": rows}

    @app.get("/activity")
    def activity(actor=Depends(identity)):
        return {"items": app.state.store.activity(actor)}

    @app.post("/feedback/{query_id}")
    def feedback(query_id: str, value: Feedback, actor=Depends(identity)):
        try:
            app.state.store.feedback(actor, query_id, value.useful, value.note)
        except KeyError:
            raise HTTPException(404, "Query not found") from None
        return {"saved": True}

    @app.delete("/documents/{document_id}")
    def delete(document_id: str, actor=Depends(identity)):
        if actor.role != "reviewer":
            raise HTTPException(403, "Reviewer required for deletion")
        return {"deleted": app.state.store.delete(actor.tenant, document_id)}

    @app.post("/search")
    def retrieve(question: Question, actor=Depends(identity)):
        started = time.perf_counter()
        docs = app.state.store.visible(actor.tenant, actor.role)
        if question.method in {"bm25", "dense", "fusion", "rerank"}:
            if question.method != "bm25" and os.getenv("EVIDENCE_NEURAL") != "1":
                raise HTTPException(
                    503, "Install the research extra and launch with --neural to enable this method"
                )
            hits = (
                app.state.indices.get(actor.tenant, actor.role, docs).rank(
                    question.question, question.method, question.limit
                )
                if docs
                else []
            )
            latency = (time.perf_counter() - started) * 1000
            query_id = app.state.store.record_query(
                actor, question.question, question.method, hits, latency
            )
            return {
                "hits": hits,
                "query_id": query_id,
                "latency_ms": latency,
                "snapshot_documents": len(docs),
                "method": question.method,
            }
        if question.method == "transformer":
            if not os.getenv("PGVECTOR_DSN"):
                raise HTTPException(503, "PGVECTOR_DSN is not configured")
            return {
                "hits": search_dense(
                    app.state.store,
                    actor.tenant,
                    actor.role,
                    question.question,
                    os.environ["PGVECTOR_DSN"],
                )
            }
        return {
            "hits": search(
                app.state.store.visible(actor.tenant, actor.role),
                question.question,
                question.method,
            )
        }

    @app.post("/vectors/reindex")
    def reindex(actor=Depends(identity)):
        if actor.role != "reviewer":
            raise HTTPException(403, "Reviewer required")
        if not os.getenv("PGVECTOR_DSN"):
            raise HTTPException(503, "PGVECTOR_DSN is not configured")
        return {
            "indexed": index_visible(
                app.state.store, actor.tenant, actor.role, os.environ["PGVECTOR_DSN"]
            )
        }

    @app.post("/answer")
    def respond(question: Question, actor=Depends(identity)):
        retrieval = None
        if question.method in {"bm25", "dense", "fusion", "rerank"}:
            retrieval = retrieve(question, actor)
            result = answer_from_hits(retrieval["hits"])
            result.update(
                {"query_id": retrieval["query_id"], "latency_ms": retrieval["latency_ms"]}
            )
        elif question.method == "transformer":
            hits = retrieve(question, actor)["hits"]
            result = answer_from_hits([hit for hit in hits if hit["score"] >= 0.25])
            result["threshold_note"] = (
                "Cosine threshold 0.25 is a demo heuristic, not calibrated answer confidence"
            )
        else:
            result = answer(
                app.state.store.visible(actor.tenant, actor.role),
                question.question,
                question.method,
            )
        if question.generate:
            if not os.getenv("ROUTER_URL"):
                raise HTTPException(503, "Generation is not configured")
            import json

            mapping = json.loads(os.getenv("SERVICE_KEYS_JSON", "{}"))
            router_key = mapping.get(actor.tenant, {}).get("router")
            if not router_key and actor.tenant == os.getenv("SERVICE_TENANT"):
                router_key = os.getenv("ROUTER_API_KEY")
            if not router_key:
                raise HTTPException(503, "No generation credential configured for this tenant")
            result = generate(result, question.question, os.environ["ROUTER_URL"], router_key)
        return result

    @app.post("/graph/path")
    def find_path(query: GraphQuery, actor=Depends(identity)):
        return {
            "path": graph_path(
                app.state.store.visible(actor.tenant, actor.role),
                query.source,
                query.target,
                query.max_hops,
            )
        }

    from .webapp import mount

    mount(app)

    return app


app = create_app()
