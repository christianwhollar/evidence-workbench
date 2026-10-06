import json
import os
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def mount(app):
    root = Path(__file__).parent
    app.mount("/assets", StaticFiles(directory=root / "web"), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(root / "web/index.html")

    @app.get("/app-config")
    def config():
        return {
            "demo": os.getenv("APP_DEMO") == "1",
            "neural": os.getenv("EVIDENCE_NEURAL") == "1",
            "generation": bool(os.getenv("ROUTER_URL")),
        }

    @app.get("/benchmarks/scifact")
    def benchmark():
        path = root / "resources/scifact.json"
        return json.loads(path.read_text()) if path.exists() else {"summary": {}}
