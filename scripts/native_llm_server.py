#!/usr/bin/env python3
"""Native OpenVINO LLM server (for Windows / Intel AI PC).

Runs Qwen3 on the GPU/NPU/CPU via openvino-genai and exposes an OpenAI-compatible
/v1/chat/completions endpoint -- so the existing gateway / OpenVINO Agent node can
call it UNCHANGED (just point the URL here).

Why it exists: OVMS in a WSL2 container can't reach the Intel GPU on Windows. This
server runs natively, so it DOES reach the chip. Same role as the old ovms-llm
service, different implementation -- and no OVMS / container in the way.

Setup (PowerShell, in the folder that contains the model dir):
    pip install openvino-genai
    # model already downloaded earlier, e.g. qwen3-8b-ov or qwen3-4b-ov
    python native_llm_server.py --model qwen3-8b-ov --device GPU --port 8001

Quick test (new PowerShell window):
    curl http://localhost:8001/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"What is an invoice? /no_think\"}],\"max_tokens\":80}"
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import openvino_genai as og

PIPE = None
DEVICE = "GPU"


def flatten(messages):
    """Collapse OpenAI chat messages into a single prompt string."""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"[System]\n{content}")
        elif role == "assistant":
            parts.append(f"[Assistant]\n{content}")
        else:
            parts.append(content)
    return "\n".join(parts)


class Handler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health" or self.path.startswith("/v1/models"):
            self._json(200, {"data": [{"id": "qwen3-ov", "object": "model"}], "object": "list"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/v1/chat/completions"):
            self._json(404, {"error": "unknown endpoint"})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except Exception as e:
            self._json(400, {"error": f"bad json: {e}"})
            return

        messages = req.get("messages", [])
        max_tokens = int(req.get("max_tokens", 512))
        prompt = flatten(messages)

        cfg = og.GenerationConfig()
        cfg.max_new_tokens = max_tokens
        temp = req.get("temperature")
        if temp is not None:
            try:
                cfg.temperature = float(temp)
                cfg.do_sample = float(temp) > 0
            except Exception:
                pass

        t0 = time.time()
        try:
            text = str(PIPE.generate(prompt, cfg))
        except Exception as e:
            self._json(500, {"error": f"generation failed: {e}"})
            return
        dt = round(time.time() - t0, 2)
        print(f"  [{DEVICE}] {len(messages)} msg -> {len(text)} chars in {dt}s")

        self._json(200, {
            "id": "chatcmpl-native",
            "object": "chat.completion",
            "model": req.get("model", "qwen3-ov"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {},
        })

    def log_message(self, *args):
        pass  # silence default request logging


def main():
    global PIPE, DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-8b-ov", help="path to the OpenVINO model dir")
    ap.add_argument("--device", default="GPU", help="GPU | NPU | CPU")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()
    DEVICE = args.device

    print(f"Loading '{args.model}' on {args.device} ... (first load can take a minute)")
    PIPE = og.LLMPipeline(args.model, args.device)
    print(f"Ready -> OpenAI-compatible endpoint at http://localhost:{args.port}/v1/chat/completions")
    print("Press Ctrl-C to stop.")
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
