# n8n-nodes-openvino

**No-code AI workflows that run entirely on local Intel hardware — CPU, GPU, or NPU. No cloud, no API keys, no data leaving your machine.**

Custom [n8n](https://n8n.io) nodes for [OpenVINO™ Model Server](https://docs.openvino.ai/2026/model-server/ovms_what_is_openvino_model_server.html), letting anyone drag-and-drop local AI pipelines that run on Intel CPU / GPU / NPU.

> **Google Summer of Code 2026** project for **OpenVINO (Intel)**
> Contributor: Nand Kishore R · Mentors: Praveen Kundurthy & Max Domeika

---

## Why

| Problem with cloud LLMs | What local OpenVINO solves |
|---|---|
| Per-token cost | Zero marginal cost after model download |
| Sensitive documents sent to third parties | Data never leaves the machine |
| Rate limits, outages, vendor lock-in | Deterministic local inference |
| Requires writing code | Drag-and-drop in n8n |

---

## What's in the box

**Two custom nodes:**

- **OpenVINO Model Server** — Predict · Document Inference · Embeddings · Chat Completion · List Models · Get Model Status. Dual transport (**REST** through a gateway, or **gRPC** direct to OVMS), AUTO-device selection (CPU/GPU/NPU) and performance hints.
- **OpenVINO Agent** *(in progress)* — an agentic loop with pluggable Chat Model + Tools + Memory, for the document pipeline.

**The target workflow — Smart Document Processing Pipeline:**

```
File Trigger → OCR (PP-OCRv5) → OpenVINO Agent → Switch → Write / Email / Skip
                                  (Qwen3-4B + 7 tools + Qdrant memory)
```

A drop-in folder where any PDF/image is read, validated, and routed by an on-device AI agent — fully offline.

---

## Quick start

Requires [Podman](https://podman.io) + `podman-compose`.

```bash
git clone https://github.com/Nandkishore-04/n8n-nodes-openvino
cd n8n-nodes-openvino

# build the custom nodes
npm install && npm run build

# configure and launch the stack
cd deployment
cp .env.example .env          # set POSTGRES_PASSWORD, N8N_ENCRYPTION_KEY, RENDER_GROUP_ID
podman-compose up -d

# n8n → http://localhost:5678
```

The stack: OVMS (classic + LLM), a Python gateway, Postgres, Qdrant, and n8n — six containers, all local.

### Models

Model IRs live under `deployment/models/` and are **gitignored** (too large for git). Fetch them:

```bash
# enhancement model (Open Model Zoo)
omz_downloader --name text-image-super-resolution-0001 -o /tmp/omz
mkdir -p deployment/models/text-sr/1
cp /tmp/omz/intel/text-image-super-resolution-0001/FP32/text-image-super-resolution-0001.{xml,bin} deployment/models/text-sr/1/
# (rename to model.xml / model.bin)
```

PP-OCRv5, DistilBERT, and BGE IRs come pre-converted (see `docs/` for conversion steps). The LLM
(`OpenVINO/Qwen3-4B-int4-ov`) auto-downloads on first `ovms-llm` start.

---

## Status

**Done (Weeks 1–2):**
- ✅ OpenVINO Model Server node — REST + gRPC transport, user-selectable
- ✅ Full 6-container Podman stack
- ✅ Clear in-UI error handling + AUTO-device plugin (device + performance hints)
- ✅ 45 automated tests at 91% coverage, ESLint, production hardening (pinned images, secrets in `.env`)

**In progress (Weeks 3–6):**
- 🔄 Device benchmarking (CPU/GPU/NPU) — see [docs/benchmarks.md](docs/benchmarks.md)
- 🔄 PP-OCRv5 document inference + PDF support
- 🔄 OpenVINO Agent node — agentic loop, built-in tools, Qdrant memory
- 🔄 Smart Document Processing Pipeline (midterm goal)

**Phase 2:** RAG chatbot over processed documents · multimodal analysis.

---

## Tech stack

| Layer | Choice |
|---|---|
| Inference runtime | OpenVINO 2026.1 / OVMS |
| LLM | Qwen3-4B-int4 (OpenVINO) |
| OCR | PP-OCRv5 |
| Embeddings | BGE-small-en-v1.5 |
| Vector DB | Qdrant |
| Workflow engine | n8n |
| Containers | Podman + podman-compose |
| Node | TypeScript 5.9, `@grpc/grpc-js` |

---

## Development

```bash
npm run dev               # tsc --watch
npm test                  # unit tests (Jest)
npm run test:integration  # integration tests (needs the stack running)
npm run lint              # eslint-plugin-n8n-nodes-base
scripts/benchmark.sh      # device benchmark (CPU/GPU/NPU)
```

---

## License

[Apache-2.0](LICENSE)
