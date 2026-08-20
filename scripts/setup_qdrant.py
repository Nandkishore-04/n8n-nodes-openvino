#!/usr/bin/env python3
"""One-time Qdrant setup for the document pipeline.

Run this ONCE before processing your first document. It creates:

  1. the `documents` collection (768-dim, Cosine) that the pipeline stores vectors in.
     Qdrant does NOT create collections automatically, so without this the pipeline's
     first upsert fails.

  2. a full-text index on the chunk `text` field. The Q&A workflow searches with two
     arms - meaning (vectors) and exact words (full text). Without this index the
     word-matching arm silently returns nothing: you get no error, just worse answers
     on things like invoice numbers and IDs.

Usage:
    python scripts/setup_qdrant.py
    python scripts/setup_qdrant.py --qdrant http://localhost:6333 --collection documents

Safe to re-run: existing collections are left alone unless you pass --recreate.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

VECTOR_SIZE = 768          # BGE-base-en-v1.5 output size
DISTANCE = "Cosine"        # BGE vectors are normalized, so cosine is the right metric


def _req(method, url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode() or "{}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant", default="http://localhost:6333")
    ap.add_argument("--collection", default="documents")
    ap.add_argument("--recreate", action="store_true",
                    help="delete and rebuild the collection (DESTROYS stored vectors)")
    a = ap.parse_args()
    base = a.qdrant.rstrip("/")
    url = f"{base}/collections/{a.collection}"

    try:
        _req("GET", f"{base}/collections")
    except Exception as e:
        print(f"Cannot reach Qdrant at {base} ({e}).", file=sys.stderr)
        print("Start Qdrant first, then run this again.", file=sys.stderr)
        return 1

    exists = True
    try:
        _req("GET", url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            exists = False
        else:
            raise

    if exists and a.recreate:
        _req("DELETE", url)
        exists = False
        print(f"deleted existing collection '{a.collection}'")

    if exists:
        print(f"collection '{a.collection}' already exists - leaving it alone")
    else:
        _req("PUT", url, {"vectors": {"size": VECTOR_SIZE, "distance": DISTANCE}})
        print(f"created collection '{a.collection}' ({VECTOR_SIZE}-dim, {DISTANCE})")

    # Full-text index on the chunk text. Re-running is harmless; Qdrant returns an error
    # if it already exists, which is not a failure worth stopping for.
    try:
        _req("PUT", f"{url}/index", {
            "field_name": "text",
            "field_schema": {"type": "text", "tokenizer": "word",
                             "lowercase": True, "min_token_len": 2},
        })
        print("created full-text index on 'text'")
    except urllib.error.HTTPError:
        print("full-text index on 'text' already present")

    info = _req("GET", url)
    points = info.get("result", {}).get("points_count", 0)
    print(f"\nReady. '{a.collection}' holds {points} chunks.")
    print("Next: start the gateway and n8n, then drop a document into your incoming folder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
