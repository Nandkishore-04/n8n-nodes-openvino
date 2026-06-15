"""
OVMS Gateway Service

Sits between n8n and the real OpenVINO Model Server.
Handles text preprocessing (tokenization) and result interpretation,
so n8n users can send plain text and get human-readable results.

Architecture:
  n8n  -->  Gateway (port 8000)  -->  Real OVMS (port 9001)
       text                     tensors               logits
       <--  readable result  <--  raw inference  <--
"""

import json
import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import numpy as np
import requests
from transformers import AutoTokenizer

# ─── Configuration ────────────────────────────────────────────────────────────

OVMS_URL = os.environ.get("OVMS_URL", "http://localhost:9001")
OVMS_LLM_URL = os.environ.get("OVMS_LLM_URL", "http://ovms-llm:8000")
OVMS_EMB_URL = os.environ.get("OVMS_EMB_URL", "http://ovms-embeddings:8000")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "/models/tokenizer")
PORT = int(os.environ.get("GATEWAY_PORT", "8000"))

# ─── Model Registry ──────────────────────────────────────────────────────────
# Maps model names to their tokenizer and label config.
# This gateway knows HOW to preprocess for each model.

MODELS = {
    "text-classifier": {
        "tokenizer_path": TOKENIZER_PATH,
        "labels": ["NEGATIVE", "POSITIVE"],
        "max_length": 512,
        "description": "DistilBERT text sentiment classifier (OpenVINO IR)",
    },
}

# ─── Globals ──────────────────────────────────────────────────────────────────

tokenizers = {}


def load_tokenizers():
    for model_name, config in MODELS.items():
        path = config["tokenizer_path"]
        print(f"  Loading tokenizer for '{model_name}' from {path}...")
        try:
            tokenizers[model_name] = AutoTokenizer.from_pretrained(path)
            print(f"  OK")
        except Exception as e:
            print(f"  FAILED: {e}")


def warm_up():
    """Fire one dummy inference per model so the first real request isn't slow.
    Best-effort: never blocks startup if OVMS isn't ready yet."""
    for model_name in MODELS:
        if model_name not in tokenizers:
            continue
        try:
            tokens = tokenizers[model_name]("warmup", return_tensors="np", padding=True, truncation=True)
            payload = {"instances": [{
                "input_ids": tokens["input_ids"].tolist()[0],
                "attention_mask": tokens["attention_mask"].tolist()[0],
            }]}
            requests.post(f"{OVMS_URL}/v1/models/{model_name}:predict", json=payload, timeout=10)
            print(f"  Warmed up '{model_name}'")
        except Exception as e:
            print(f"  Warm-up skipped for '{model_name}' (OVMS not ready yet): {e}")


def _aggregate_confidence(pages):
    """Text-length-weighted mean confidence across PDF pages (text-layer pages = 1.0)."""
    total_len = sum(len(p["text"]) for p in pages)
    if total_len == 0:
        return 0.0
    weighted = sum(p.get("confidence", 0.0) * len(p["text"]) for p in pages)
    return round(weighted / total_len, 4)


# ─── HTTP Handler ─────────────────────────────────────────────────────────────

class GatewayHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        path = urlparse(self.path).path

        # Health check
        if path == "/health" or path == "/":
            self.send_json({
                "status": "healthy",
                "ovms_url": OVMS_URL,
                "models": list(MODELS.keys()),
                "tokenizers_loaded": list(tokenizers.keys()),
            })
            return

        # Proxy model listing to OVMS
        if path == "/v1/config":
            try:
                resp = requests.get(f"{OVMS_URL}/v1/config", timeout=5)
                self.send_json(resp.json())
            except Exception as e:
                self.send_error_json(502, f"Cannot reach OVMS: {e}")
            return

        # Proxy /v1/models to OVMS-LLM /v3/models (for n8n OpenAI Chat Model compatibility)
        if path == "/v1/models":
            try:
                resp = requests.get(f"{OVMS_LLM_URL}/v3/models", timeout=10)
                self.send_json(resp.json())
            except Exception as e:
                self.send_error_json(502, f"Cannot reach OVMS-LLM: {e}")
            return

        # Proxy model status to OVMS
        if path.startswith("/v1/models/"):
            try:
                resp = requests.get(f"{OVMS_URL}{path}", timeout=5)
                self.send_json(resp.json())
            except Exception as e:
                self.send_error_json(502, f"Cannot reach OVMS: {e}")
            return

        self.send_error_json(404, f"Unknown endpoint: {path}")

    def do_POST(self):
        path = urlparse(self.path).path
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"

        try:
            request_data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error_json(400, "Invalid JSON")
            return

        # Proxy /v1/chat/completions → OVMS-LLM /v3/chat/completions
        # (n8n's OpenAI Chat Model node calls /v1/, but OVMS uses /v3/)
        if path == "/v1/chat/completions":
            try:
                resp = requests.post(
                    f"{OVMS_LLM_URL}/v3/chat/completions",
                    json=request_data,
                    headers={"Content-Type": "application/json"},
                    # Qwen3 on CPU is ~7 tok/s; large contexts (multi-page docs) can take minutes.
                    timeout=int(os.environ.get("LLM_PROXY_TIMEOUT", "600")),
                )
                self.send_response(resp.status_code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(resp.content)
            except Exception as e:
                self.send_error_json(502, f"OVMS-LLM proxy failed: {e}")
            return

        # POST /v1/responses — shim for the OpenAI "Responses API" (n8n's built-in agent uses it).
        # OVMS only speaks chat/completions, so translate request → chat, response → responses shape.
        if path == "/v1/responses":
            self.handle_responses(request_data)
            return

        # POST /v1/embeddings → OVMS-Embeddings /v3/embeddings (BGE, OpenAI-compatible) for RAG
        if path == "/v1/embeddings":
            try:
                resp = requests.post(
                    f"{OVMS_EMB_URL}/v3/embeddings",
                    json=request_data,
                    headers={"Content-Type": "application/json"},
                    timeout=60,
                )
                self.send_response(resp.status_code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(resp.content)
            except Exception as e:
                self.send_error_json(502, f"Embeddings proxy failed: {e}")
            return

        # POST /v1/document/infer — PDF/image → PP-OCRv5 → {text, boxes, confidence}
        if path == "/v1/document/infer":
            self.handle_document(request_data)
            return

        device_hint = self.headers.get("X-Target-Device", "AUTO")

        # POST /v2/models/{name}/infer — KServe v2
        if "/v2/models/" in path and path.endswith("/infer"):
            model_name = path.split("/v2/models/")[1].split("/")[0]
            self.handle_inference(model_name, request_data, device_hint, "v2")
            return

        # POST /v1/models/{name}:predict — TF Serving v1
        if "/v1/models/" in path and ":predict" in path:
            model_name = path.split("/v1/models/")[1].split(":predict")[0]
            self.handle_inference(model_name, request_data, device_hint, "v1")
            return

        self.send_error_json(404, f"Unknown endpoint: {path}")

    def _messages_from_responses(self, data):
        """Build chat messages from an OpenAI Responses-API request (string | message list)."""
        messages = []
        instr = data.get("instructions")
        if instr:
            messages.append({"role": "system", "content": str(instr)})
        inp = data.get("input")
        if isinstance(inp, str):
            messages.append({"role": "user", "content": inp})
        elif isinstance(inp, list):
            for item in inp:
                if isinstance(item, str):
                    messages.append({"role": "user", "content": item})
                elif isinstance(item, dict):
                    role = item.get("role", "user")
                    if role == "developer":
                        role = "system"
                    content = item.get("content")
                    if isinstance(content, list):
                        text = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
                    else:
                        text = str(content)
                    messages.append({"role": role, "content": text})
        return messages

    def handle_responses(self, data):
        """OpenAI Responses API shim → chat/completions on OVMS-LLM, response re-shaped to Responses."""
        messages = self._messages_from_responses(data)
        body = {
            "model": data.get("model"),
            "messages": messages,
            "temperature": data.get("temperature", 0.7),
            "max_tokens": data.get("max_output_tokens", data.get("max_tokens", 512)),
        }
        try:
            r = requests.post(
                f"{OVMS_LLM_URL}/v3/chat/completions", json=body,
                timeout=int(os.environ.get("LLM_PROXY_TIMEOUT", "600")),
            )
            cc = r.json()
        except Exception as e:
            self.send_error_json(502, f"responses shim failed: {e}")
            return

        import re
        content = cc.get("choices", [{}])[0].get("message", {}).get("content", "")
        content = re.sub(r"^\s*<think>[\s\S]*?</think>\s*", "", content).strip()
        usage = cc.get("usage", {})
        self.send_json({
            "id": f"resp_{int(time.time() * 1000)}",
            "object": "response",
            "created_at": int(time.time()),
            "model": cc.get("model", data.get("model")),
            "status": "completed",
            "output": [{
                "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": content, "annotations": []}],
            }],
            "output_text": content,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        })
        print(f"  -> /v1/responses shim → {usage.get('total_tokens', 0)} tokens")

    def handle_document(self, request_data):
        """PDF/image (base64) → PP-OCRv5 → {text, boxes, confidence}.
        Body: { data: <base64>, filename?: str, dpi?: int }
        """
        import base64
        b64 = request_data.get("data") or request_data.get("image_b64")
        if not b64:
            self.send_error_json(400, "Send base64 file bytes in 'data'.")
            return
        try:
            raw = base64.b64decode(b64)
        except Exception as e:
            self.send_error_json(400, f"Invalid base64: {e}")
            return

        filename = (request_data.get("filename") or "").lower()
        # OCR fallback renders at a higher DPI than 150 — dense text needs it to stay legible.
        ocr_dpi = int(request_data.get("dpi", 200))
        is_pdf = filename.endswith(".pdf") or raw[:5] == b"%PDF-"
        # A page with at least this many embedded characters is treated as a digital
        # (text-layer) page — extracted directly. Below it, the page is assumed scanned → OCR.
        MIN_TEXT_LAYER_CHARS = 20

        try:
            from models.ppocr import run_ocr
            import numpy as np
            import cv2
        except Exception as e:
            self.send_error_json(503, f"OCR module unavailable: {e}")
            return

        start = time.time()
        try:
            if is_pdf:
                import fitz  # PyMuPDF
                doc = fitz.open(stream=raw, filetype="pdf")
                pages = []
                for page in doc:
                    # Hybrid: prefer the embedded text layer (digital PDF) — instant & exact.
                    embedded = page.get_text().strip()
                    if len(embedded) >= MIN_TEXT_LAYER_CHARS:
                        pages.append({"text": embedded, "source": "text-layer", "confidence": 1.0, "boxes": []})
                        continue
                    # No usable text layer → scanned page → OCR at higher DPI.
                    try:
                        pix = page.get_pixmap(dpi=ocr_dpi)
                        arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
                        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if pix.n >= 3 else cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                        p = run_ocr(bgr)
                        p["source"] = "ocr"
                        pages.append(p)
                    except Exception as e:  # isolate per-page failure
                        pages.append({"text": "", "source": "ocr-failed", "confidence": 0.0, "boxes": [], "error": str(e)})

                result = {
                    "pages": pages,
                    "text": "\n\n".join(p["text"] for p in pages if p["text"].strip()),
                    "page_count": len(pages),
                    "confidence": _aggregate_confidence(pages),
                    "source": "text-layer" if all(p.get("source") == "text-layer" for p in pages) else "mixed",
                }
            else:
                arr = np.frombuffer(raw, np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    self.send_error_json(400, "Could not decode image bytes.")
                    return
                result = run_ocr(bgr)
                result["source"] = "ocr"
        except Exception as e:
            self.send_error_json(502, f"OCR failed: {e}")
            return

        result["inference_time_ms"] = round((time.time() - start) * 1000, 2)
        self.send_json(result)
        n = result.get("page_count", len(result.get("boxes", [])))
        print(f"  -> document [{result.get('source')}]: {n} page(s)/region(s), "
              f"conf={result.get('confidence')} in {result['inference_time_ms']:.0f}ms")

    def handle_inference(self, model_name, request_data, device_hint, api_version):
        # Check if we have a tokenizer for this model
        if model_name not in MODELS:
            self.send_error_json(404, f"Model '{model_name}' not registered in gateway")
            return

        if model_name not in tokenizers:
            self.send_error_json(503, f"Tokenizer for '{model_name}' not loaded")
            return

        config = MODELS[model_name]
        tokenizer = tokenizers[model_name]

        # Extract text from the request
        text = self.extract_text(request_data, api_version)
        if text is None:
            self.send_error_json(400, "Could not extract text from request. Send JSON with a 'text' field.")
            return

        start_time = time.time()

        # Tokenize
        tokens = tokenizer(
            text,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=config["max_length"],
        )

        # Send tokenized data to real OVMS
        ovms_payload = {
            "instances": [{
                "input_ids": tokens["input_ids"].tolist()[0],
                "attention_mask": tokens["attention_mask"].tolist()[0],
            }]
        }

        try:
            ovms_resp = requests.post(
                f"{OVMS_URL}/v1/models/{model_name}:predict",
                json=ovms_payload,
                timeout=30,
            )
            ovms_resp.raise_for_status()
            ovms_result = ovms_resp.json()
        except requests.exceptions.RequestException as e:
            self.send_error_json(502, f"OVMS inference failed: {e}")
            return

        total_time_ms = (time.time() - start_time) * 1000

        # Interpret the logits
        logits = np.array(ovms_result["predictions"][0])
        probs = self.softmax(logits)
        labels = config["labels"]
        predicted_idx = int(np.argmax(probs))

        result = {
            "input_text": text,
            "label": labels[predicted_idx],
            "confidence": round(float(probs[predicted_idx]), 4),
            "scores": {labels[i]: round(float(probs[i]), 4) for i in range(len(labels))},
            "actual_device": device_hint,
            "inference_time_ms": round(total_time_ms, 2),
            "model": model_name,
        }

        # Format response based on API version
        if api_version == "v2":
            response = {
                "model_name": model_name,
                "model_version": "1",
                "outputs": [{
                    "name": "output",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [json.dumps(result)],
                }],
                "actual_device": device_hint,
                "inference_time_ms": round(total_time_ms, 2),
            }
        else:
            response = {
                "predictions": [result],
                "model_name": model_name,
                "model_version": "1",
                "actual_device": device_hint,
                "inference_time_ms": round(total_time_ms, 2),
            }

        self.send_json(response)
        print(f"  -> \"{text[:60]}...\" => {result['label']} ({result['confidence']:.2%}) in {total_time_ms:.0f}ms")

    def extract_text(self, request_data, api_version):
        """Extract plain text from various request formats."""
        try:
            if api_version == "v2":
                input_data = request_data.get("inputs", [{}])[0].get("data", [""])[0]
                try:
                    parsed = json.loads(input_data)
                    if isinstance(parsed, dict):
                        return parsed.get("text", parsed.get("document_text", parsed.get("prompt", str(parsed))))
                    return str(parsed)
                except (json.JSONDecodeError, TypeError):
                    return str(input_data) if input_data else None
            else:
                # Accept {"instances": [{"text": "..."}]} (TF Serving format)
                instances = request_data.get("instances", [])
                if instances:
                    instance = instances[0]
                    if isinstance(instance, str):
                        return instance
                    return instance.get("text", instance.get("document_text", instance.get("prompt")))
                # Also accept {"text": "..."} directly (simple format from n8n node)
                return request_data.get("text", request_data.get("document_text", request_data.get("prompt")))
        except Exception:
            return None

    def softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))

    def send_error_json(self, status, message):
        self.send_json({"error": message, "status": status}, status=status)

    def log_message(self, format, *args):
        print(f"[Gateway] {args[0]}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("")
    print("  +--------------------------------------------------+")
    print("  |   OVMS Gateway Service                           |")
    print(f"  |   Gateway:  http://localhost:{PORT}                |")
    print(f"  |   OVMS:     {OVMS_URL:<36} |")
    print("  +--------------------------------------------------+")
    print("")

    load_tokenizers()
    print("")
    print("  Warming up models...")
    warm_up()
    print("")

    print(f"  Gateway ready on port {PORT}")
    print(f"  n8n should connect to: http://gateway:{PORT}")
    print("")

    server = HTTPServer(("0.0.0.0", PORT), GatewayHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down gateway...")
        server.shutdown()


if __name__ == "__main__":
    main()
