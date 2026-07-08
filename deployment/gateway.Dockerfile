# Container image for the OpenVINO gateway (scripts/native_gateway.py) — VLM OCR, Qwen3 LLM,
# CLIP document triage, BGE embeddings — all via openvino-genai / optimum-intel.
#
# NATIVE run (python scripts/native_gateway.py) is the TESTED path and works on Windows and Linux.
# This image is provided for Linux users who prefer containers, on an Intel host with GPU drivers.
# It defaults every stage to GPU (well-supported in containers). NPU-in-container triage is possible
# but best-effort: pass /dev/accel and set CLIP_DEVICE=NPU (see podman-compose.yml comments).
#
# Build context is the REPO ROOT (so it can COPY scripts/ + deployment/).
FROM openvino/ubuntu24_dev:2025.0.0
USER root
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY deployment/requirements.gateway.txt .
RUN pip install --no-cache-dir -r requirements.gateway.txt

COPY scripts/native_gateway.py .

EXPOSE 8000
# Model dirs are mounted at /models (see compose). Flags/devices are supplied by podman-compose.yml.
ENTRYPOINT ["python3", "-u", "native_gateway.py"]
CMD ["--models", "/models", "--host", "0.0.0.0", "--port", "8000", "--ocr-engine", "vlm"]
