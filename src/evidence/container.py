"""Container demo startup. Production deployments use the unseeded API entry point."""

import json
import os
from pathlib import Path


def main():
    from .store import Store, Document

    directory = Path(os.getenv("DATA_DIR", "/data"))
    store = Store(directory / "evidence.db")
    marker = directory / ".seeded-v2"
    if os.getenv("APP_DEMO") == "1" and not marker.exists():
        for item in json.loads((Path(__file__).parent / "resources/operations.json").read_text()):
            store.ingest("alpha", Document(**item))
        marker.write_text("synthetic operations corpus v2\n")
    import uvicorn

    uvicorn.run("evidence.api:app", host="0.0.0.0", port=8102)


if __name__ == "__main__":
    main()
