import os
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


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=3000)
    method: Literal["bm25", "latent", "hybrid", "transformer"] = "hybrid"
    generate: bool = False


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

    @app.delete("/documents/{document_id}")
    def delete(document_id: str, actor=Depends(identity)):
        if actor.role != "reviewer":
            raise HTTPException(403, "Reviewer required for deletion")
        return {"deleted": app.state.store.delete(actor.tenant, document_id)}

    @app.post("/search")
    def retrieve(question: Question, actor=Depends(identity)):
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
        if question.method == "transformer":
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
            if not os.getenv("ROUTER_URL") or not os.getenv("ROUTER_API_KEY"):
                raise HTTPException(503, "Generation is not configured")
            result = generate(
                result, question.question, os.environ["ROUTER_URL"], os.environ["ROUTER_API_KEY"]
            )
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

    return app


app = create_app()
