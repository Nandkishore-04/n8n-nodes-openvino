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
                    timeout=120,
                )
                self.send_response(resp.status_code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(resp.content)
            except Exception as e:
                self.send_error_json(502, f"OVMS-LLM proxy failed: {e}")
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
                instances = request_data.get("instances", [{}])
                instance = instances[0] if instances else {}
                if isinstance(instance, str):
                    return instance
                return instance.get("text", instance.get("document_text", instance.get("prompt", None)))
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
