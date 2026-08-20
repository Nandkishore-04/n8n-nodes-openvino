# Deploying WF1 on the Lunar Lake AI PC

Goal: run the **exact same Podman stack** that works locally, but on the Intel Lunar Lake
(Core Ultra Series 2) instance, so OCR + LLM run on the **Intel GPU** at usable speed — and
swap the LLM up to **Qwen3-8B** for stronger reasoning.

## The one hard constraint (read first)

The Tiber AI Cloud Lunar Lake instance is **Windows-only**. Linux containers run via **WSL2**.
WSL2 has a **GPU tunnel** (GPU-PV → `/dev/dri`) but **no NPU tunnel** — so:

- ✅ **GPU** is reachable inside containers → OCR + LLM + embeddings run on the iGPU/Arc GPU.
- ❌ **NPU** (`/dev/accel`) is **not** reachable from a WSL2 container on Windows.

So the containerized workflow targets the **GPU**. The NPU stays a **native benchmark story**
(the W3 numbers), not part of the containerized pipeline. Set expectations accordingly: this
deployment proves "fast local AI on Intel GPU," not "pipeline on NPU."

---

## Phase A — Prepare the Windows instance

1. **Connect** to the Lunar Lake instance (RDP from the Tiber AI Cloud console).
2. **Install WSL2 + Ubuntu** (PowerShell as admin):
   ```powershell
   wsl --install -d Ubuntu
   ```
   Reboot if prompted; set the Ubuntu username/password on first launch.
3. **Intel GPU driver for WSL** — install the latest Intel Arc/iGPU driver on *Windows*
   (Intel's WSL-enabled driver exposes the GPU to WSL2 via `/dev/dri`). No driver is
   installed *inside* WSL.
4. **Verify GPU passthrough** in the Ubuntu (WSL2) shell:
   ```bash
   ls -l /dev/dri          # expect card0 + renderD128
   sudo apt install -y clinfo && clinfo | grep -i "Device Name"   # should list the Intel GPU
   ```
   If `/dev/dri` is missing, the Windows GPU driver isn't WSL-enabled — fix that before continuing.

## Phase B — Install tooling inside WSL2 (Ubuntu)

```bash
sudo apt update && sudo apt -y upgrade
sudo apt install -y podman podman-compose git curl build-essential
# Node 20 (for the custom node build + website)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - && sudo apt install -y nodejs
```

## Phase C — Get the project + assets

1. **Clone the repo** into the WSL2 home (not `/mnt/c` — keep it on the Linux filesystem for speed):
   ```bash
   cd ~ && git clone <your-repo-url> n8n-nodes-openvino && cd n8n-nodes-openvino
   ```
2. **Copy the pre-converted model IRs** — `deployment/models/*/` is **gitignored** (too large for
   git), so it won't come from the clone. Transfer it separately (from this laptop):
   ```bash
   # on the laptop:
   tar czf models.tgz -C deployment models
   # move models.tgz to the Lunar Lake (RDP copy / scp / cloud storage), then in WSL2:
   tar xzf models.tgz -C deployment/
   ```
   The LLM + embeddings models **auto-download from HuggingFace** on first run — only the
   OCR/classifier IRs need copying.
3. **Create `deployment/.env`** (not in git):
   ```
   POSTGRES_PASSWORD=n8npassword
   N8N_ENCRYPTION_KEY=<generate: openssl rand -hex 16>
   RENDER_GROUP_ID=<see Phase D step 1>
   ```
4. **Build the custom node** (n8n mounts `dist/`):
   ```bash
   npm install && npm run build
   ```
5. **Recreate the watch folders** + open perms (rootless Podman uid-mapping):
   ```bash
   mkdir -p ~/proj-demo/{incoming,processing,processed,failed}
   chmod -R 777 ~/proj-demo
   ```
   Update the `proj-demo` mount path in `deployment/podman-compose.yml` to the new location
   (`/home/<user>/proj-demo:/data/proj-demo`), and `WATCH_DIR` in `web/.env.local` likewise.

## Phase D — Chip navigation + 8B configuration

**Device plan** (WSL2 container can reach GPU + CPU; NPU is NOT tunneled into containers):

| Model | Device | Why |
|---|---|---|
| Qwen3-8B (LLM) | GPU | heaviest compute |
| ppocr-det, ppocr-rec | GPU | OCR CNNs |
| text-sr | GPU | super-res CNN |
| BGE embeddings | CPU | light — keep off the GPU so it doesn't contend with the LLM |
| text-classifier | CPU | light, barely used in WF1 |

1. **Render group id** — containers need the host's `render` group to use `/dev/dri`:
   ```bash
   getent group render | cut -d: -f3      # put this number in .env as RENDER_GROUP_ID
   ```
2. **OCR/super-res → GPU** — mount the GPU-targeted config instead of the default. In the
   `ovms` service of `deployment/podman-compose.yml`, change the config mount:
   ```yaml
   volumes:
     - ./config.lunarlake.json:/config.json:ro    # was ./config.json — GPU targets for OCR/text-sr
     - ./models:/models:ro
   ```
   (`config.lunarlake.json` sets `target_device: GPU` for ppocr-det/rec + text-sr, CPU for the
   classifier. If a GPU load errors, switch that model back to `AUTO` for a safe CPU fallback.)
3. **LLM → 8B on GPU** in the `ovms-llm` service:
   ```yaml
   command: >
     --source_model OpenVINO/Qwen3-8B-int4-ov   # was Qwen3-4B-int4-ov
     --model_repository_path /models
     --task text_generation
     --target_device GPU                          # run the LLM on the GPU
     --rest_port 8000
     --cache_size 2
   mem_limit: 10g    # was 6g — 8B int4 needs more headroom
   ```
   And set the **OpenVINO Agent** node's **LLM Model** = `OpenVINO/Qwen3-8B-int4-ov`.
   ⚠️ Verify `--target_device GPU` is honored in the OVMS-LLM logs on first boot (OVMS GenAI device
   handling varies by build). If it's rejected, drop the flag (defaults to CPU) and confirm the
   device in the logs, or set it via the generated graph config — validate on-device.
4. **Embeddings → CPU** — `ovms-embeddings` defaults to CPU; leave it (keeps the GPU for the LLM).
5. **Auto-create the metadata tables on first boot** — on a fresh Lunar Lake the postgres volume is
   empty, so mount the init script so it runs automatically (add to the `postgres` service):
   ```yaml
   volumes:
     - postgres_data:/var/lib/postgresql/data
     - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
   ```
   (On the existing laptop the volume already has data, so we ran it manually instead.)

### Verify the chips are actually used (do this on first boot)
```bash
# OCR/super-res models — confirm they loaded on GPU, not CPU:
podman logs ovms 2>&1 | grep -iE "device|GPU|loaded"
# LLM — confirm the device it bound to:
podman logs ovms-llm 2>&1 | grep -iE "device|GPU|AVAILABLE"
# Sanity: a doc should now process in SECONDS, not minutes. If OCR/LLM still say CPU,
# /dev/dri isn't reaching the container (recheck Phase A step 4).
```

## Phase E — Launch + verify

1. **Bring up the stack** (first run downloads the 8B model, ~5 GB — be patient):
   ```bash
   cd deployment && podman-compose up -d
   podman logs -f ovms-llm        # wait for "state changed to: AVAILABLE"
   ```
2. **Confirm the GPU is actually used** — OVMS logs should show a GPU device being loaded
   (not CPU). A single agent call should now take **seconds**, not minutes.
3. **Run the website**:
   ```bash
   cd ../web && npm install && npm run dev    # http://localhost:3000
   ```
   WSL2 forwards `localhost` to Windows, so open `localhost:3000` / `:5678` in the **Windows browser**.
4. **In n8n** (`localhost:5678`): import `workflows/smart-document-pipeline.json`, create the
   `Pipeline Postgres` credential (host `postgres`, db `n8n`, user `n8n`, pass from `.env`),
   assign it to the 6 Postgres nodes, (optional SMTP), then **Activate**.
5. **End-to-end test**: drop a PDF on the website → it should process **fast** on the GPU →
   Documents tab fills → re-drop the same file → dedup-skipped.

## What "done" looks like

- A document processes in **seconds** (vs minutes on the laptop iGPU).
- The 8B model gives visibly better extraction/reasoning than 4B.
- Same dedup / metadata / audit / dashboard behaviour as the verified local run.

## Honest gaps to mention to mentors

- **NPU** isn't in the containerized path (WSL2 limitation) — it's covered by the native W3 benchmarks.
- **GPU** is the workhorse here; that's the realistic "Intel AI PC" demo for a containerized stack.
- Exposing Postgres `:5432` and the dev servers is fine for a single-user demo box; not a
  multi-tenant production posture.
