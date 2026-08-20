#!/usr/bin/env python3
"""Seed the agent's reference knowledge base (Qdrant 'knowledge_base' collection).

Each entry is a short reference snippet the agent retrieves via `knowledge_search` BEFORE
extracting -- document-type playbooks + business rules + domain facts. Fully local: text is
embedded with BGE via the gateway, stored in Qdrant. No cloud.

Usage:
  python scripts/seed_knowledge.py --gateway http://127.0.0.1:8000 --qdrant http://127.0.0.1:6333
  (laptop/containers: --gateway http://localhost:8000 --qdrant http://localhost:6333)
"""
import argparse
import json
import urllib.error
import urllib.request

EMB_MODEL = "OpenVINO/bge-base-en-v1.5-int8-ov"  # native gateway ignores the name; OVMS uses it

KNOWLEDGE = [
    {"topic": "invoice", "text": "Invoice playbook: extract vendor/seller name, invoice number, invoice date, line items (description, quantity, unit price, amount), subtotal, tax, and total. Validate subtotal + tax = total. Flag any invoice with total over 10000 for human approval. Use GSTIN / tax id as the vendor identifier."},
    {"topic": "receipt", "text": "Receipt playbook: extract merchant name, date, items with prices, and the total paid. Categorize the expense (food, travel, office). Receipts under 100 need no approval."},
    {"topic": "resume", "text": "Resume / CV playbook: extract candidate name, contact (email, phone), education, skills, and work experience with dates. Classify seniority (intern, junior, mid, senior) from years of experience."},
    {"topic": "contract", "text": "Contract playbook: extract the parties, effective date, term/duration, payment terms, and key obligations. Flag any contract missing a signature or an effective date for review."},
    {"topic": "purchase order", "text": "Purchase order playbook: extract PO number, buyer, supplier, line items, and total. When possible, match the PO number against existing invoices."},
    {"topic": "business rules", "text": "Business rules: a document whose file hash already exists is a duplicate and must be skipped. Documents older than 90 days are flagged as stale. Any monetary total above 10000 requires human approval."},
    {"topic": "tax rates india", "text": "India GST reference: GST is 5%, 12%, 18%, or 28% by category, usually split equally into CGST and SGST (18% = 9% + 9%). Restaurant service is commonly 5%."},
]


def _post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def _put(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}, method="PUT")
    return urllib.request.urlopen(req, timeout=60).read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default="http://127.0.0.1:8000")
    ap.add_argument("--qdrant", default="http://127.0.0.1:6333")
    ap.add_argument("--collection", default="knowledge_base")
    a = ap.parse_args()

    # create the collection (768-dim, Cosine); 409 = already exists, which is fine
    try:
        _put(f"{a.qdrant}/collections/{a.collection}", {"vectors": {"size": 768, "distance": "Cosine"}})
    except urllib.error.HTTPError as e:
        if e.code != 409:
            raise

    points = []
    for i, k in enumerate(KNOWLEDGE):
        emb = _post(f"{a.gateway}/v1/embeddings", {"model": EMB_MODEL, "input": k["text"]})
        points.append({"id": i + 1, "vector": emb["data"][0]["embedding"], "payload": {"topic": k["topic"], "text": k["text"]}})
        print(f"  embedded [{k['topic']}]")
    _put(f"{a.qdrant}/collections/{a.collection}/points", {"points": points})
    print(f"Seeded {len(points)} entries into '{a.collection}'.")


if __name__ == "__main__":
    main()
