import argparse
import httpx
from evidence.demo import DOCUMENTS

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8102")
args = parser.parse_args()
with httpx.Client(headers={"Authorization": "Bearer demo-reviewer"}, timeout=30) as client:
    for document in DOCUMENTS:
        response = client.post(args.url.rstrip("/") + "/documents", json=document)
        response.raise_for_status()
        print(response.json())
