import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Launch the evidence browser and retrieval service"
    )
    parser.add_argument("--demo", action="store_true")
    parser.add_argument(
        "--neural",
        action="store_true",
        help="Enable pinned dense and reranker models; first use downloads weights",
    )
    parser.add_argument("--port", type=int, default=8102)
    parser.add_argument("--data-dir", default="runtime")
    args = parser.parse_args()
    os.environ["DATA_DIR"] = args.data_dir
    if args.neural:
        os.environ["EVIDENCE_NEURAL"] = "1"
        import torch

        torch.set_num_threads(4)
    if args.demo:
        os.environ["APP_DEMO"] = "1"
        from .store import Store, Document

        store = Store(Path(args.data_dir) / "evidence.db")
        # Seed once: deliberately revoked or edited documents stay changed on restart.
        marker = Path(args.data_dir) / ".seeded-v2"
        if not marker.exists():
            for item in json.loads(
                (Path(__file__).parent / "resources/operations.json").read_text()
            ):
                store.ingest("alpha", Document(**item))
            marker.write_text("synthetic operations corpus v2\n")
    import uvicorn

    uvicorn.run("evidence.api:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
