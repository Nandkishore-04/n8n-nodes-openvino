<div align="center">

<img src="assets/openvino-logo.png" alt="OpenVINO" height="56"/> &nbsp;&nbsp;&nbsp;&nbsp; <img src="assets/gsoc-logo.png" alt="Google Summer of Code" height="56"/>

# n8n-nodes-openvino

**Agentic AI document workflows that run entirely on your Intel AI PC — CPU, GPU, or NPU. No cloud, no API keys, no data leaving your machine.**

Custom [n8n](https://n8n.io) nodes + a native [OpenVINO™](https://docs.openvino.ai) gateway that turn any PDF or image into structured, validated, searchable data — with an on-device AI agent doing the reasoning.

> **Google Summer of Code 2026** · **OpenVINO (Intel)**
> Contributor: Nand Kishore R · Mentors: Praveen Kundurthy & Max Domeika

</div>

---

## What it does — Smart Document Processing

Drop a document in a folder (or upload it), and it flows through a fully local pipeline:

```
File → dedup (sha256) → 🔍 CLIP triage (NPU) → OCR (VLM, GPU) → 🤖 Agent extract + validate → Postgres + Qdrant
                             │                                        │
                        not a document?                          enriched / flagged / duplicate
                             ↓                                        ↓
                     rejected/ (recoverable)                   processed/  •  searchable
```

- **CLIP triage on the NPU** — a real vision transformer decides *"is this a processable document?"* in ~20 ms before any heavy work; photos/selfies are rejected (recoverably), never silently dropped.
- **VLM OCR on the GPU** — Qwen2.5-VL reads layout, tables, and degraded scans near-perfectly; digital PDFs use the exact text layer.
- **On-device agent** — Qwen3-8B extracts a schema it picks per document type, then a deterministic validation layer checks the math (line-items → subtotal → tax → total, amount-in-words, coverage).
- **Stored + searchable** — queryable metadata in Postgres, embeddings in Qdrant, full audit trail.

Every model runs on **Intel silicon via OpenVINO** — the right chip for each job (NPU for the glance, GPU for accuracy).

---

## Requirements

- **Intel AI PC** (Core Ultra / Lunar Lake / Meteor Lake) with GPU + NPU — or an Intel Linux box with GPU. *(macOS is CPU-only and not recommended; the value is the Intel GPU/NPU.)*
- **Python 3.10+**, **Node.js 20+**
- **Postgres** and **Qdrant** (local)
- OpenVINO model files (see [Models](#1-models))

> **Why native, not containers?** On **Windows**, Docker/Podman run inside a WSL2 Linux VM that can't pass the Intel NPU/GPU through — so the gateway runs **natively**. The native path also works on **Linux**. (A best-effort Podman stack for Linux lives in `deployment/` — see [Containers](#containers-linux-experimental).)

---

## Setup

### 0. Build the nodes
```bash
git clone https://github.com/Nandkishore-04/n8n-nodes-openvino
cd n8n-nodes-openvino
npm install && npm run build      # compiles the custom nodes to dist/
```

### 1. Models
Place OpenVINO IR model directories where the gateway can find them (e.g. `deployment/models/`):

| Model | Purpose | How to get it |
|---|---|---|
| **Qwen2.5-VL-7B** (OpenVINO) | OCR (GPU) | download the OpenVINO build from Hugging Face |
| **Qwen3-8B-int4-ov** | Agent LLM (GPU) | `OpenVINO/Qwen3-8B-int4-ov` on Hugging Face |
| **bge-base-en-v1.5-int8-ov** | Embeddings (CPU) | `OpenVINO/bge-base-en-v1.5-int8-ov` (auto-loads via optimum) |
| **CLIP** (ViT-B/32) | Triage (NPU) | convert it yourself → `python scripts/convert_clip.py` |

```bash
pip install openvino-genai "optimum[openvino]" opencv-python pymupdf numpy torch transformers
python scripts/convert_clip.py     # writes deployment/models/clip/
```

### 2. Folders
Create the pipeline I/O folders (this is your `docRoot`):
```bash
# Linux
mkdir -p /data/proj-demo/{incoming,processing,processed,failed,rejected}
# Windows (PowerShell)
mkdir C:\Users\you\proj-demo\incoming, C:\Users\you\proj-demo\processing, C:\Users\you\proj-demo\processed, C:\Users\you\proj-demo\failed, C:\Users\you\proj-demo\rejected
```

### 3. Postgres + Qdrant
```bash
# apply the pipeline schema once
psql -h localhost -U <user> -d <db> -f deployment/sql/init.sql
# Qdrant: run the qdrant binary (or container) so it listens on :6333
```

### 4. Start the gateway (native — the tested path)
This is the one process that talks to the chips. **Classify on NPU, OCR + LLM on GPU:**
```bash
python scripts/native_gateway.py \
  --models deployment/models \
  --ocr-engine vlm --vlm-model deployment/models/qwen2.5-vl-7b \
  --llm <path-to>/qwen3-8b-ov \
  --ocr-device GPU --llm-device GPU --clip-device NPU \
  --port 8000
```
Wait for `Ready -> http://127.0.0.1:8000` and `CLIP document triage -> NPU`. By default it binds **127.0.0.1** (local only); add `--host 0.0.0.0 --api-key <secret>` only if you need remote access.

*(No GPU/NPU? Swap in `--ocr-device CPU --clip-device CPU` — slower, but it runs.)*

### 5. Start n8n with the custom nodes
Set the env the pipeline needs, then launch n8n:
```bash
# Linux
export N8N_CUSTOM_EXTENSIONS="$PWD/dist"
export NODE_FUNCTION_ALLOW_BUILTIN="fs,crypto"
export N8N_RESTRICT_FILE_ACCESS_TO="/data/proj-demo"
n8n
```
```powershell
# Windows (PowerShell) — same session, then run n8n
$env:N8N_CUSTOM_EXTENSIONS="C:\...\n8n-nodes-openvino\dist"
$env:NODE_FUNCTION_ALLOW_BUILTIN="fs,crypto"
$env:N8N_RESTRICT_FILE_ACCESS_TO="C:/Users/you/proj-demo"
n8n
```
n8n → http://localhost:5678

### 6. Import + configure the workflow
1. **Import** `workflows/smart-document-pipeline.json`
2. Open the **`Config`** node → set `docRoot` to your folder, and `gatewayUrl` / `qdrantUrl` if they differ
3. Create two **credentials** (n8n prompts on import):
   - **Postgres** — your DB host/user/password
   - **OpenVINO Model Server** — Gateway URL `http://127.0.0.1:8000`
4. **Activate** the workflow

### 7. Run it
Drop a PDF/image into `incoming/`. Within seconds:
- a **document** → OCR'd, extracted, validated → `processed/` + rows in Postgres + vectors in Qdrant
- a **photo/selfie** → rejected on the NPU glance → `rejected/` (visible, recoverable)
- a **duplicate** → skipped at the hash gate

---

## The custom nodes

- **OpenVINO Model Server** — `Classify Document` (CLIP triage on NPU), `Document Inference` (VLM OCR), `Embeddings`, `Chat Completion`, `Predict`, model status. Target device selectable (CPU/GPU/NPU/AUTO).
- **OpenVINO Agent** — a local ReAct loop over the document text with built-in tools (extract fields, validate totals, coverage check, flag for review, dedup, recall).

## Device layout (recommended)

| Stage | Chip | Why |
|---|---|---|
| Document triage (CLIP) | **NPU** | tiny, fast, the right chip for a cheap "glance" |
| OCR (Qwen2.5-VL) | **GPU** | accuracy on layout/tables/degraded scans |
| Agent (Qwen3-8B) | **GPU** | throughput for the reasoning loop |
| Embeddings (BGE) | CPU | light, runs anywhere |

## Containers (Linux, experimental)

A Podman stack (`deployment/podman-compose.yml` + `gateway.Dockerfile`) is provided for Linux users who prefer containers — Intel device passthrough works on native Linux. **It is not yet verified end-to-end** (native is the tested path); GPU is the default, NPU-in-container is best-effort. See the comments in `deployment/podman-compose.yml`.

## Development
```bash
npm run dev     # tsc --watch
npm test        # unit tests (Jest)
npm run lint    # eslint-plugin-n8n-nodes-base
```

## License
[Apache-2.0](LICENSE)
