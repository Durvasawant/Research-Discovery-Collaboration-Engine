"""How ANOTHER group consumes this asset (e.g. Group 8 Placement & Career Intelligence, Group 4 Knowledge Assistant).
Uses only the standard library.

    python -m src.api.simple_server --port 8000 &        # or: uvicorn src.api.app:app
    python demo/client_example.py
"""
import json, os, urllib.request

BASE = os.environ.get("RDCE_URL", "http://127.0.0.1:8000")


def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


if __name__ == "__main__":
    print(json.load(urllib.request.urlopen(BASE + "/health")))
    r = post("/find-related-researchers", {"faculty_id": "F05", "k": 3})
    for x in r["result"]["related_researchers"]:
        print(f'{x["faculty_name"]}  sim={x["similarity"]}  shared={x["shared_keywords"][:3]}')
    print("common topics:", r["result"]["common_topics"])
    # Group 8 style integration: turn a job description into a list of matching researchers / skills
    s = post("/search-researchers", {"text": "Looking for candidates with Python, NLP, transformers and cloud", "k": 3})
    print([(x["faculty_id"], x["score"]) for x in s["results"]])
    print(post("/extract-entities", {"text": "Fine-tune BERT for entity linking on a knowledge graph"})["result"])
