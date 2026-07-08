#!/usr/bin/env python3
"""Native OpenVINO gateway for Windows / Intel AI PC (no OVMS, no containers).

Runs the whole inference layer directly on the chips and exposes the SAME HTTP API the
n8n nodes already call, so nothing in the workflow changes -- just point the credential's
Gateway URL here.

  POST /v1/document/infer    PDF/image (base64) -> OCR text + confidence   (OCR on GPU/NPU)
  POST /v1/embeddings        text -> BGE vector                            (embeddings)
  POST /v1/chat/completions  OpenAI-compatible                            (Qwen3 on GPU)
  GET  /health

Why: OVMS in a WSL2 container can't reach the Intel GPU on Windows. Native OpenVINO can.
This is the same idea as native_llm_server.py, extended to OCR + embeddings.

Setup (PowerShell):
  pip install openvino openvino-genai "optimum[openvino]" opencv-python pymupdf numpy
  python native_gateway.py --models <repo>\\deployment\\models --llm qwen3-8b-ov ^
      --ocr-device GPU --llm-device GPU --port 8000

`--models` must contain: ppocr-det\\1\\model.xml, ppocr-rec\\1\\model.xml, ppocr-v5-en-dict.txt
"""
import argparse
import base64
import json
import math
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import openvino as ov

# ---- globals (set in main) ------------------------------------------------
DET = REC = None          # compiled OpenVINO OCR models
CTC_CHARS = []            # rec vocab
LLM = None                # openvino_genai pipeline
LLM_NAME = "qwen3-ov"     # id reported by /v1/models (for n8n's OpenAI Chat Model node)
EMBED = None              # (tokenizer, model) for BGE, or None
OCR_DEVICE = "GPU"
DET_STATIC = None         # (H, W) when the detector has a fixed input (NPU); else None
OCR_ENGINE = "ppocr"      # "ppocr" (DB, GPU/CPU) or "omz" (OMZ det+rec, NPU-capable)
HTD = None                # OMZ horizontal-text-detection-0001 (compiled)
RTR = None                # OMZ text-recognition-0014 (compiled)
OMZ_ALPHABET = "#0123456789abcdefghijklmnopqrstuvwxyz"  # index 0 = CTC blank
VLM = None                # openvino_genai VLMPipeline (Qwen2.5-VL document OCR), or None
VLM_MODEL_DIR = None      # path to the VLM model (so OCR can be recompiled on another chip)
MODELS_DIR = None         # --models root (for recompiling OMZ / PP-OCR on another chip)
DET_DIR = REC_DIR = None  # PP-OCR det/rec subdir names
CURRENT_OCR_DEVICE = None # the chip the OCR model is currently compiled on
SR = None                 # BSRGAN super-resolution (compiled), or None when not installed
SR_MAX_SIDE = 1024        # cap the long side before 4x SR to bound compute/memory
CLIP = None               # compiled CLIP image encoder for NPU document triage, or None
CLIP_TEXT_EMB = None      # precomputed normalized label text embeddings [L, D]
CLIP_IS_DOC = None        # [L] 1/0 — which labels count as "document"
CLIP_LABELS = None        # [L] label strings (for logging the best match)
CLIP_LOGIT_SCALE = 100.0  # CLIP temperature (exp(logit_scale)) from the export
CLIP_DEVICE = None        # the chip CLIP is compiled on (NPU by default)
API_KEY = ""              # if set, every request must send Authorization: Bearer <API_KEY>
INFER_LOCK = threading.Lock()  # OV infer requests are not concurrency-safe; serialize all inference

# ===========================================================================
# OCR  (PP-OCRv5 preprocessing copied verbatim from gateway/models/ppocr.py)
# ===========================================================================
def _infer(compiled, tensor):
    return list(compiled(tensor).values())[0]


def _preprocess_det(img, limit=960):
    h, w = img.shape[:2]
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    if DET_STATIC:
        # NPU path: letterbox into the fixed square (resize preserving aspect, pad
        # right/bottom). Boxes map back by dividing by `scale`, so ratio = 1/scale.
        SH, SW = DET_STATIC
        scale = min(SH / h, SW / w)
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        canvas = np.zeros((SH, SW, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
        x = ((canvas.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)[None]
        return x.astype(np.float32), (1.0 / scale, 1.0 / scale)
    # dynamic path (GPU/CPU): resize longest side to `limit`, round to /32
    scale = limit / max(h, w)
    nh, nw = int(round(h * scale / 32) * 32), int(round(w * scale / 32) * 32)
    x = (cv2.resize(img, (nw, nh)).astype(np.float32) / 255.0)
    x = ((x - mean) / std).transpose(2, 0, 1)[None]
    return x.astype(np.float32), (h / nh, w / nw)


def _reading_order(boxes):
    """Order boxes the way a person reads: cluster into rows by vertical position, then sort
    left-to-right within each row. Keeps table cells aligned with their column headers, so the
    LLM can map values correctly (a flat top-to-bottom sort scrambles multi-column tables)."""
    bs = [b for b in boxes if (float(b[3]) - float(b[1])) > 1]
    if not bs:
        return list(boxes)
    rh = max(8.0, float(np.median([float(b[3]) - float(b[1]) for b in bs])))
    bs.sort(key=lambda b: float(b[1]))
    rows, cur, cur_y = [], [], None
    for b in bs:
        cy = (float(b[1]) + float(b[3])) / 2.0
        if cur_y is not None and abs(cy - cur_y) > rh * 0.6:
            rows.append(cur)
            cur = []
        cur.append(b)
        cur_y = cy
    if cur:
        rows.append(cur)
    out = []
    for row in rows:
        out.extend(sorted(row, key=lambda b: float(b[0])))
    return out


def _db_boxes(prob_map, ratio, thresh=0.3, box_thresh=0.5):
    rh, rw = ratio
    binary = (prob_map > thresh).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < 20:
            continue
        x, y, w, h = cv2.boundingRect(c)
        conf = float(prob_map[y:y + h, x:x + w].mean())
        if conf < box_thresh:
            continue
        pad = 3
        boxes.append((max(0, int(x * rw) - pad), max(0, int(y * rh) - pad),
                      int((x + w) * rw) + pad, int((y + h) * rh) + pad, conf))
    return _reading_order(boxes)


def _preprocess_rec(crop, h=48, max_w=320):
    ch, cw = crop.shape[:2]
    nw = min(max_w, max(1, int(cw * h / ch)))
    resized = cv2.resize(crop, (nw, h))
    x = ((resized.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)
    padded = np.zeros((3, h, max_w), np.float32)
    padded[:, :, :nw] = x
    return padded[None].astype(np.float32)


def _ctc_decode(logits):
    seq = logits[0].argmax(axis=1)
    text, prev = [], -1
    for idx in seq:
        if idx != 0 and idx != prev and idx < len(CTC_CHARS):
            text.append(CTC_CHARS[idx])
        prev = idx
    return "".join(text)


# ===========================================================================
# OMZ OCR  (horizontal-text-detection-0001 + text-recognition-0014) -- NPU-capable
# ===========================================================================
def _recognize_omz(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 32))             # model input is (W=128, H=32)
    inp = resized[None, None].astype(np.float32)       # [1, 1, 32, 128]
    logits = np.squeeze(_infer(RTR, inp), axis=1)      # [16, 1, 37] -> [16, 37]
    seq = logits.argmax(axis=1)
    out, prev = [], -1
    for idx in seq:
        if idx != 0 and idx != prev:                   # 0 = CTC blank
            out.append(OMZ_ALPHABET[idx])
        prev = idx
    return "".join(out)


def run_ocr_omz(img):
    """OMZ OCR: detection (static 704x704) + recognition (static 32x128). Both NPU-capable.
    Note: the recognizer is alphanumeric-only (no case / punctuation)."""
    h, w = img.shape[:2]
    inp = cv2.resize(img, (704, 704)).transpose(2, 0, 1)[None].astype(np.float32)  # [1,3,704,704] BGR
    res = HTD(inp)
    boxes = np.squeeze([v for v in res.values() if v.ndim >= 2 and v.shape[-1] == 5][0])  # [100,5]
    boxes = _reading_order([b for b in boxes if float(b[4]) >= 0.15])  # row-major reading order
    sx, sy = w / 704.0, h / 704.0
    lines, confs = [], []
    pad = 2
    for b in boxes:
        x1, y1, x2, y2, conf = (float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4]))
        if conf < 0.15:                                # lower filter -> catch more text regions
            continue
        X1, Y1 = max(0, int(x1 * sx) - pad), max(0, int(y1 * sy) - pad)
        X2, Y2 = min(w, int(x2 * sx) + pad), min(h, int(y2 * sy) + pad)
        crop = img[Y1:Y2, X1:X2]
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            continue
        txt = _recognize_omz(crop)
        if txt.strip():
            lines.append(txt)
            confs.append(conf)
    conf = round(float(np.mean(confs)), 4) if confs else 0.0
    return {"text": "\n".join(lines), "confidence": 0.0 if math.isnan(conf) else conf,
            "confidence_source": "detector"}


# ===========================================================================
# VLM OCR  (Qwen2.5-VL via openvino_genai) -- reads the whole document holistically,
# preserving layout/tables. Far more robust than PP-OCR on screenshots/photos. GPU or NPU.
# ===========================================================================
VLM_PROMPT = ("Transcribe ALL text in this document image exactly as written. Preserve the reading "
              "order and table layout: keep each table row's cells together, left to right, on one line. "
              "CRITICAL: transcribe ONLY what is clearly legible — never guess, invent, auto-correct, or "
              "translate text. If a character, number, date or word is not clearly readable, write [?] in "
              "its place instead of guessing. Output only the transcribed text, with no commentary.")
# NOTE: document triage (is this a processable document?) no longer lives in this prompt — it runs
# as a dedicated CLIP zero-shot classifier on the NPU (classify_clip), BEFORE this OCR pass.


def _vlm_confidence(res, text):
    """Real confidence from the VLM's own token log-probs (geometric-mean token probability),
    so an uncertain read drops below the review threshold instead of always reporting 1.0.
    Returns (confidence, source) where source is 'logprobs' (real, from token scores) or
    'markers' (fallback [?]-heuristic when this openvino-genai build doesn't expose scores) —
    so it's visible at a glance whether real per-token confidence is actually flowing."""
    try:
        # openvino-genai exposes per-sequence cumulative log-prob as res.scores across builds;
        # newer builds also carry it on perf_metrics. Try both before giving up.
        scores = getattr(res, "scores", None)
        cum_logprob = None
        if scores:
            cum_logprob = float(scores[0])
        else:
            pm = getattr(res, "perf_metrics", None)
            raw = getattr(pm, "scores", None) if pm is not None else None
            if raw:
                cum_logprob = float(raw[0])
        if cum_logprob is not None:
            toks = getattr(res, "tokens", None)
            ntok = len(toks[0]) if toks else max(1, round(len(text) / 4))  # ~4 chars/token
            conf = math.exp(cum_logprob / max(1, ntok))      # geometric-mean token probability
            if 0.0 < conf <= 1.0:
                return round(conf, 4), "logprobs"
    except Exception:
        pass
    # fallback: confidence from the [?] illegible markers the prompt asks the VLM to emit
    unreadable = text.count("[?]")
    words = max(1, len(text.split()))
    return round(max(0.0, 1.0 - unreadable / words), 4), "markers"


def run_ocr_vlm(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = ov.Tensor(np.ascontiguousarray(rgb[None].astype(np.uint8)))  # [1,H,W,3] RGB
    try:  # ask for token log-probs; not all genai builds accept the kwarg
        res = VLM.generate(VLM_PROMPT, images=[tensor], max_new_tokens=2048, do_sample=False, logprobs=1)
    except Exception:
        res = VLM.generate(VLM_PROMPT, images=[tensor], max_new_tokens=2048, do_sample=False)
    text = str(res).strip()
    conf, source = _vlm_confidence(res, text)
    return {"text": text, "confidence": conf, "confidence_source": source}


# ===========================================================================
# Document triage — CLIP zero-shot classifier on the NPU. A real ViT on the NPU (the deliverable),
# deterministic (no prompt oscillation), with a genuine confidence score. Runs BEFORE OCR: image ->
# NPU image embedding -> cosine-sim vs the precomputed doc / non-doc label embeddings.
# ===========================================================================
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def _clip_preprocess(bgr):
    """CLIP preprocessing: resize shortest side to 224, center-crop 224, RGB, normalize -> [1,3,224,224]."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    s = 224.0 / min(h, w)
    rgb = cv2.resize(rgb, (max(224, round(w * s)), max(224, round(h * s))), interpolation=cv2.INTER_CUBIC)
    h, w = rgb.shape[:2]
    top, left = (h - 224) // 2, (w - 224) // 2
    rgb = rgb[top:top + 224, left:left + 224]
    x = (rgb.astype(np.float32) / 255.0 - _CLIP_MEAN) / _CLIP_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])  # [1,3,224,224]


def classify_clip(bgr):
    """Zero-shot doc/non-doc verdict from the CLIP image embedding vs the cached label embeddings.
    Returns is_document + a real confidence (P(document)). If CLIP isn't loaded it errs toward
    processing — never silently drop a real document."""
    if CLIP is None or CLIP_TEXT_EMB is None:
        return {"is_document": True, "worth_processing": True, "confidence": 1.0, "doc_type": "other",
                "reason": "CLIP classifier not loaded; defaulting to process", "source": "clip-unavailable"}
    emb = np.asarray(CLIP(_clip_preprocess(bgr))[CLIP.output(0)][0], dtype=np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    logits = CLIP_LOGIT_SCALE * (CLIP_TEXT_EMB @ emb)       # cosine-sim * scale, one per label
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    doc_prob = float(probs[CLIP_IS_DOC == 1].sum())         # total probability mass on document labels
    best = int(np.argmax(probs))
    return {"is_document": bool(doc_prob >= 0.5), "worth_processing": bool(doc_prob >= 0.5),
            "confidence": round(doc_prob, 4), "doc_type": "other",
            "best_label": str(CLIP_LABELS[best]) if CLIP_LABELS is not None else "",
            "source": f"clip-{(CLIP_DEVICE or 'cpu').lower()}"}


def run_ocr(img):
    if OCR_ENGINE == "vlm":
        return run_ocr_vlm(img)
    if OCR_ENGINE == "omz":
        return run_ocr_omz(img)
    x, ratio = _preprocess_det(img)
    prob = _infer(DET, x)[0, 0]
    boxes = _db_boxes(prob, ratio)
    lines, total_area, weighted = [], 0.0, 0.0
    for (x1, y1, x2, y2, det_conf) in boxes:
        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            continue
        text = _ctc_decode(_infer(REC, _preprocess_rec(crop)))
        if not text.strip():
            continue
        area = float((x2 - x1) * (y2 - y1))
        total_area += area
        weighted += det_conf * area
        lines.append(text)
    confidence = round(weighted / total_area, 4) if total_area else 0.0
    return {"text": "\n".join(lines), "confidence": 0.0 if math.isnan(confidence) else confidence,
            "confidence_source": "detector"}


# ===========================================================================
# BSRGAN super-resolution (the agent's enhance / re-OCR lever)  -- GPU/CPU, not NPU
# ===========================================================================
def _super_resolve(bgr):
    """4x super-resolve a BGR image with BSRGAN, returning a sharper BGR image.
    Large inputs are capped on the long side first so the 4x output stays bounded."""
    h, w = bgr.shape[:2]
    long_side = max(h, w)
    if long_side > SR_MAX_SIDE:
        s = SR_MAX_SIDE / float(long_side)
        bgr = cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = rgb.transpose(2, 0, 1)[None]                     # [1,3,H,W] RGB 0..1
    out = list(SR(inp).values())[0]                        # [1,3,4H,4W]
    out = np.clip(out[0].transpose(1, 2, 0), 0.0, 1.0)     # HWC 0..1
    return cv2.cvtColor((out * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


def _ocr_maybe_enhance(bgr, enhance):
    """OCR the image; if enhance is requested and BSRGAN is loaded, also OCR the
    super-resolved image and keep whichever reads better (higher confidence, longer text as
    tie-break). SR can add artifacts, so the guard means enhancement only helps or is neutral."""
    base = run_ocr(bgr)
    if not (enhance and SR is not None):
        return base
    try:
        enh = run_ocr(_super_resolve(bgr))
        bc, ec = base.get("confidence"), enh.get("confidence")
        bc = bc if isinstance(bc, (int, float)) and not math.isnan(bc) else None
        ec = ec if isinstance(ec, (int, float)) and not math.isnan(ec) else None
        if bc is not None and ec is not None:
            better = ec > bc or (ec == bc and len(enh.get("text", "")) > len(base.get("text", "")))
        else:
            # confidence unusable (e.g. NaN) -> fall back to how much text each read
            better = len(enh.get("text", "").strip()) > len(base.get("text", "").strip())
        if better:
            enh["enhanced"] = True
            return enh
    except Exception as e:
        print(f"  BSRGAN enhance failed, using original image: {e}")
    base["enhanced"] = False
    return base


def _ensure_ocr_device(device):
    """Switch the OCR chip on request (n8n's Target Device). Recompiles the active engine on the
    requested device, single-instance so memory stays bounded — the first call on a new chip pays a
    recompile. Best-effort: if a chip can't take the model (e.g. NPU + a too-big VLM), stays put."""
    global VLM, HTD, RTR, DET, REC, DET_STATIC, CURRENT_OCR_DEVICE
    if not device or device == CURRENT_OCR_DEVICE:
        return None
    try:
        core = ov.Core()
        if OCR_ENGINE == "vlm":
            import openvino_genai as og
            VLM = og.VLMPipeline(VLM_MODEL_DIR, device)
        elif OCR_ENGINE == "omz":
            HTD = core.compile_model(os.path.join(MODELS_DIR, "htd", "1", "model.xml"), device)
            RTR = core.compile_model(os.path.join(MODELS_DIR, "rtr", "1", "model.xml"), device)
        else:
            DET = core.compile_model(os.path.join(MODELS_DIR, DET_DIR, "1", "model.xml"), device)
            REC = core.compile_model(os.path.join(MODELS_DIR, REC_DIR, "1", "model.xml"), device)
            ishape = DET.input(0).partial_shape
            DET_STATIC = (ishape[2].get_length(), ishape[3].get_length()) if ishape.is_static else None
        CURRENT_OCR_DEVICE = device
        print(f"  [OCR] recompiled on {device}")
        return None
    except Exception as e:
        msg = f"chip '{device}' not usable for this model ({type(e).__name__}: {e}); ran on '{CURRENT_OCR_DEVICE}' instead"
        print(f"  [OCR] {msg}")
        return msg


def handle_document(req):
    raw = base64.b64decode(req.get("data") or req.get("image_b64"))
    filename = (req.get("filename") or "").lower()
    enhance = bool(req.get("enhance", False))
    classify = req.get("mode") == "classify"
    is_pdf = filename.endswith(".pdf") or raw[:5] == b"%PDF-"

    # Triage mode: a cheap CLIP "glance" on the NPU — is this even a processable document?
    if classify:
        if is_pdf:
            import fitz
            page = fitz.open(stream=raw, filetype="pdf")[0]
            if len(page.get_text().strip()) >= 20:  # digital text layer = certainly a document
                return {"is_document": True, "worth_processing": True, "confidence": 1.0, "doc_type": "other",
                        "reason": "digital PDF with a text layer", "source": "digital-pdf"}
            pix = page.get_pixmap(dpi=150)
            arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if pix.n >= 3 else cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        return classify_clip(bgr)

    if is_pdf:
        import fitz
        doc = fitz.open(stream=raw, filetype="pdf")
        pages, dpi = [], int(req.get("dpi", 200))
        for page in doc:
            embedded = page.get_text().strip()
            if len(embedded) >= 20:  # digital text layer = read it directly, no OCR
                pages.append({"text": embedded, "confidence": 1.0, "confidence_source": "digital-pdf"})
                continue
            pix = page.get_pixmap(dpi=dpi)
            arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if pix.n >= 3 else cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            pages.append(_ocr_maybe_enhance(bgr, enhance))
        total = sum(len(p["text"]) for p in pages) or 1
        conf = round(sum(p["confidence"] * len(p["text"]) for p in pages) / total, 4)
        # report the source of the weakest page — that's the one that would trigger review
        weakest = min(pages, key=lambda p: p.get("confidence", 1.0)) if pages else {}
        return {"text": "\n\n".join(p["text"] for p in pages if p["text"].strip()),
                "confidence": conf, "confidence_source": weakest.get("confidence_source", "digital-pdf"),
                "page_count": len(pages),
                "source": "pdf-enhanced" if any(p.get("enhanced") for p in pages) else "pdf"}
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    r = _ocr_maybe_enhance(bgr, enhance)
    r["source"] = "image-enhanced" if r.get("enhanced") else "image"
    return r


# ===========================================================================
# Embeddings (BGE via optimum-intel) and Chat (Qwen3 via openvino-genai)
# ===========================================================================
def embed(texts):
    tok, model = EMBED
    import torch  # optimum returns torch tensors
    enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    emb = out.last_hidden_state[:, 0]  # CLS pooling (BGE)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy().tolist()


def flatten(messages):
    parts = []
    for m in messages:
        r, c = m.get("role", "user"), m.get("content", "")
        parts.append(f"[System]\n{c}" if r == "system" else (f"[Assistant]\n{c}" if r == "assistant" else c))
    return "\n".join(parts)


def messages_from_responses(data):
    """Build chat messages from an OpenAI Responses-API request (n8n's Tools Agent uses /responses)."""
    messages = []
    if data.get("instructions"):
        messages.append({"role": "system", "content": str(data["instructions"])})
    inp = data.get("input")
    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for item in inp:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                itype = item.get("type")
                if itype == "function_call":  # the model's own prior tool call (history)
                    messages.append({"role": "assistant",
                                     "content": f'<tool_call>{{"name": "{item.get("name")}", "arguments": {item.get("arguments", "{}")}}}</tool_call>'})
                    continue
                if itype == "function_call_output":  # the tool's result coming back
                    messages.append({"role": "user", "content": f'Tool result: {item.get("output")}'})
                    continue
                role = "system" if item.get("role") == "developer" else item.get("role", "user")
                content = item.get("content")
                if isinstance(content, list):
                    text = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
                else:
                    text = str(content)
                messages.append({"role": role, "content": text})
    return messages


# ===========================================================================
# HTTP
# ===========================================================================
class Handler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def _authed(self):
        # no key configured -> allow (the default 127.0.0.1 bind already limits access to this host)
        if not API_KEY:
            return True
        return self.headers.get("Authorization", "") == f"Bearer {API_KEY}"

    def do_GET(self):
        if not self._authed():
            self._json(401, {"error": "unauthorized"})
            return
        if "models" in self.path:
            # OpenAI-compatible model list so n8n's OpenAI Chat Model node connects cleanly
            self._json(200, {"object": "list", "data": [
                {"id": LLM_NAME, "object": "model", "owned_by": "openvino"}]})
        elif self.path == "/health":
            try:
                available = ov.Core().available_devices  # which chips OpenVINO actually sees
            except Exception:
                available = []
            self._json(200, {"status": "ok", "ocr_engine": OCR_ENGINE, "ocr_device": CURRENT_OCR_DEVICE,
                             "available_devices": available, "embeddings": EMBED is not None,
                             "super_resolution": SR is not None})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self._authed():
            self._json(401, {"error": "unauthorized"})
            return
        t0 = time.time()
        try:
            req = self._read()
            # match by suffix so any base-URL convention works (/v1/chat/completions, /chat/completions, …)
            if "document/infer" in self.path:
                with INFER_LOCK:
                    # classify runs on CLIP (its own chip) — don't recompile the heavy OCR engine for it
                    warning = None if req.get("mode") == "classify" else _ensure_ocr_device(req.get("device"))
                    r = handle_document(req)
                r["inference_time_ms"] = round((time.time() - t0) * 1000, 1)
                r["device"] = CURRENT_OCR_DEVICE                       # the chip that actually ran it
                r["device_requested"] = req.get("device") or CURRENT_OCR_DEVICE
                if warning:
                    r["warning"] = warning                            # e.g. "chip 'NPU' not usable …"
                if "text" not in r:  # classify-only (CLIP triage on NPU)
                    print(f"  [TRIAGE {CLIP_DEVICE or CURRENT_OCR_DEVICE}] is_document={r.get('is_document')} conf={r.get('confidence')} ({r.get('source')}) best='{r.get('best_label', '')}' in {r['inference_time_ms']:.0f}ms")
                else:
                    print(f"  [OCR {CURRENT_OCR_DEVICE}] conf={r.get('confidence')} ({r.get('confidence_source')}) in {r['inference_time_ms']:.0f}ms{' | ' + warning if warning else ''}")
                self._json(200, r)
            elif "embeddings" in self.path:
                if EMBED is None:
                    self._json(503, {"error": "embeddings not loaded (install optimum[openvino])"})
                    return
                inp = req.get("input")
                with INFER_LOCK:
                    vecs = embed(inp if isinstance(inp, list) else [inp])
                self._json(200, {"object": "list", "data": [
                    {"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)]})
            elif "chat/completions" in self.path:
                prompt = flatten(req.get("messages", [])) + " /no_think"
                mnt = int(req.get("max_tokens", 512))
                with INFER_LOCK:
                    try:  # repetition_penalty curbs runaway repetition loops; not all builds accept the kwarg
                        text = str(LLM.generate(prompt, max_new_tokens=mnt, repetition_penalty=1.3))
                    except Exception:
                        text = str(LLM.generate(prompt, max_new_tokens=mnt))
                print(f"  [LLM] {len(text)} chars in {(time.time()-t0):.1f}s")
                self._json(200, {"id": "chatcmpl-native", "object": "chat.completion",
                                 "model": req.get("model", "qwen3-ov"),
                                 "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]})
            elif "responses" in self.path:
                msgs = messages_from_responses(req)
                tools = req.get("tools") or []
                prompt = flatten(msgs)
                if tools:  # prompt-based function calling: tell Qwen3 how to emit a tool call
                    defs = json.dumps([(t.get("function") or {"name": t.get("name"),
                                        "description": t.get("description"), "parameters": t.get("parameters")}) for t in tools])
                    prompt += ("\n\nAVAILABLE TOOLS (JSON): " + defs +
                               '\nTo call a tool, reply with ONLY: <tool_call>{"name": "<tool_name>", "arguments": {<args>}}</tool_call>'
                               " and nothing else. Once you have the tool result and can answer, reply in plain text.")
                prompt += " /no_think"
                mnt = int(req.get("max_output_tokens", req.get("max_tokens", 512)))
                with INFER_LOCK:
                    try:
                        text = str(LLM.generate(prompt, max_new_tokens=mnt, repetition_penalty=1.3))
                    except Exception:
                        text = str(LLM.generate(prompt, max_new_tokens=mnt))
                text = re.sub(r"^\s*<think>[\s\S]*?</think>\s*", "", text).strip()
                base = {"id": f"resp_{int(time.time()*1000)}", "object": "response", "created_at": int(time.time()),
                        "model": req.get("model", LLM_NAME), "status": "completed",
                        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}
                tc = re.search(r"<tool_call>([\s\S]*?)</tool_call>", text)
                call = None
                if tc:
                    raw_call = tc.group(1)
                    try:
                        call = json.loads(raw_call[raw_call.index("{"):raw_call.rindex("}") + 1])
                    except Exception:
                        # model botched the JSON (e.g. )) instead of }}) — salvage the tool name, default args
                        nm = re.search(r'"name"\s*:\s*"([^"]+)"', raw_call)
                        if nm:
                            ag = re.search(r'"arguments"\s*:\s*(\{[\s\S]*\})', raw_call)
                            args = {}
                            if ag:
                                try:
                                    args = json.loads(ag.group(1))
                                except Exception:
                                    args = {}
                            call = {"name": nm.group(1), "arguments": args}
                if call and call.get("name"):
                    cid = f"call_{int(time.time() * 1000)}"
                    base["output"] = [{"type": "function_call", "id": f"fc_{cid}", "call_id": cid,
                                       "name": call["name"], "arguments": json.dumps(call.get("arguments", {})),
                                       "status": "completed"}]
                    base["output_text"] = ""
                    print(f"  [LLM /responses] -> tool_call {call['name']}")
                else:
                    base["output"] = [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                                       "content": [{"type": "output_text", "text": text, "annotations": []}]}]
                    base["output_text"] = text
                    print(f"  [LLM /responses] {len(text)} chars in {(time.time()-t0):.1f}s")
                self._json(200, base)
            else:
                self._json(404, {"error": "unknown endpoint"})
        except Exception as e:
            self._json(500, {"error": str(e)})

    def log_message(self, *a):
        pass


def main():
    global DET, REC, CTC_CHARS, LLM, EMBED, OCR_DEVICE, DET_STATIC, OCR_ENGINE, HTD, RTR, SR, SR_MAX_SIDE, VLM
    global VLM_MODEL_DIR, MODELS_DIR, DET_DIR, REC_DIR, CURRENT_OCR_DEVICE, LLM_NAME
    global CLIP, CLIP_TEXT_EMB, CLIP_IS_DOC, CLIP_LABELS, CLIP_LOGIT_SCALE, CLIP_DEVICE, API_KEY
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="dir with ppocr-det/rec + dict (or htd/rtr for omz)")
    ap.add_argument("--llm", default="qwen3-8b-ov", help="openvino-genai model dir")
    ap.add_argument("--bge", default="OpenVINO/bge-base-en-v1.5-int8-ov", help="BGE model (HF id or local dir)")
    ap.add_argument("--ocr-device", default="GPU")
    ap.add_argument("--llm-device", default="GPU")
    ap.add_argument("--ocr-engine", default="ppocr", choices=["ppocr", "omz", "vlm"],
                    help="ppocr=PP-OCRv5; omz=OMZ det+rec (NPU); vlm=Qwen2.5-VL document OCR (GPU/NPU)")
    ap.add_argument("--vlm-model", default="", help="VLM OCR model dir (default <models>/qwen2.5-vl)")
    ap.add_argument("--static-ocr", action="store_true",
                    help="ppocr only: use the static-shape PP-OCR models")
    ap.add_argument("--sr-device", default="GPU", help="BSRGAN enhance device (GPU/CPU; not NPU)")
    ap.add_argument("--sr-model", default="", help="BSRGAN IR path (default <models>/bsrgan/1/model.xml)")
    ap.add_argument("--sr-max-side", type=int, default=1024, help="cap the long side before 4x SR")
    ap.add_argument("--clip-model", default="", help="CLIP triage IR dir (default <models>/clip); run scripts/convert_clip.py")
    ap.add_argument("--clip-device", default="NPU", help="chip for the CLIP document-triage classifier (NPU recommended)")
    ap.add_argument("--host", default="127.0.0.1", help="bind address: 127.0.0.1 = local only (secure default); 0.0.0.0 = all interfaces")
    ap.add_argument("--api-key", default=os.environ.get("GATEWAY_API_KEY", ""),
                    help="require 'Authorization: Bearer <key>' on every request (set this when binding to 0.0.0.0)")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    OCR_DEVICE = a.ocr_device
    OCR_ENGINE = a.ocr_engine
    SR_MAX_SIDE = a.sr_max_side
    MODELS_DIR = a.models
    CURRENT_OCR_DEVICE = a.ocr_device

    def _exec(compiled):  # what device AUTO actually selected
        try:
            return compiled.get_property("EXECUTION_DEVICES")
        except Exception:
            return a.ocr_device

    core = ov.Core()
    if a.ocr_engine == "vlm":
        print(f"VLM OCR (Qwen2.5-VL) requested -> {a.ocr_device}")
        import openvino_genai as og
        vlm_dir = a.vlm_model or os.path.join(a.models, "qwen2.5-vl")
        VLM_MODEL_DIR = vlm_dir
        VLM = og.VLMPipeline(vlm_dir, a.ocr_device)
        print(f"  VLM OCR loaded from {vlm_dir}")
    elif a.ocr_engine == "omz":
        print(f"OMZ OCR (htd + text-recognition-0014) requested -> {a.ocr_device}")
        HTD = core.compile_model(os.path.join(a.models, "htd", "1", "model.xml"), a.ocr_device)
        RTR = core.compile_model(os.path.join(a.models, "rtr", "1", "model.xml"), a.ocr_device)
        print(f"  OCR running on: {_exec(HTD)}")
    else:
        det_dir = "ppocr-det-static" if a.static_ocr else "ppocr-det"
        rec_dir = "ppocr-rec-static" if a.static_ocr else "ppocr-rec"
        DET_DIR, REC_DIR = det_dir, rec_dir
        print(f"OCR models ({det_dir}, {rec_dir}) requested -> {a.ocr_device}")
        DET = core.compile_model(os.path.join(a.models, det_dir, "1", "model.xml"), a.ocr_device)
        REC = core.compile_model(os.path.join(a.models, rec_dir, "1", "model.xml"), a.ocr_device)
        print(f"  OCR running on: {_exec(DET)}")
        ishape = DET.input(0).partial_shape
        if ishape.is_static:
            dims = [d.get_length() for d in ishape]
            DET_STATIC = (dims[2], dims[3])
            print(f"  static detector input {DET_STATIC} -> letterbox preprocessing")
        with open(os.path.join(a.models, "ppocr-v5-en-dict.txt"), encoding="utf-8") as f:
            CTC_CHARS = ["[blank]"] + [ln.rstrip("\n") for ln in f] + [" "]

    sr_path = a.sr_model or os.path.join(a.models, "bsrgan", "1", "model.xml")
    if os.path.exists(sr_path):
        SR = core.compile_model(sr_path, a.sr_device)
        print(f"BSRGAN super-resolution -> {a.sr_device} ({_exec(SR)})")
    else:
        print(f"BSRGAN IR not found at {sr_path} -- enhance path disabled (run scripts/convert_bsrgan.py)")

    # CLIP zero-shot document triage on the NPU (the "glance"). Graceful: if it can't load/compile,
    # triage returns 'process' so a real document is never silently dropped.
    clip_dir = a.clip_model or os.path.join(a.models, "clip")
    clip_xml = os.path.join(clip_dir, "1", "model.xml")
    clip_labels = os.path.join(clip_dir, "labels.npz")
    if os.path.exists(clip_xml) and os.path.exists(clip_labels):
        try:
            CLIP = core.compile_model(clip_xml, a.clip_device)
            CLIP_DEVICE = a.clip_device
            z = np.load(clip_labels, allow_pickle=True)
            CLIP_TEXT_EMB = z["text_emb"].astype(np.float32)
            CLIP_IS_DOC = z["is_doc"]
            CLIP_LABELS = z["labels"]
            CLIP_LOGIT_SCALE = float(z["logit_scale"]) if "logit_scale" in z else 100.0
            print(f"CLIP document triage -> {a.clip_device} ({_exec(CLIP)}), {len(CLIP_LABELS)} labels")
        except Exception as e:
            CLIP = None
            print(f"  CLIP triage failed on {a.clip_device} ({type(e).__name__}: {e}) -- triage defaults to 'process'")
    else:
        print(f"CLIP IR not found at {clip_xml} -- triage disabled, defaults to 'process' (run scripts/convert_clip.py)")

    print(f"LLM '{a.llm}' -> {a.llm_device} (first load can take a minute)")
    import openvino_genai as og
    LLM = og.LLMPipeline(a.llm, a.llm_device)
    LLM_NAME = os.path.basename(a.llm.rstrip("/\\")) or a.llm

    try:
        from optimum.intel import OVModelForFeatureExtraction
        from transformers import AutoTokenizer
        print(f"Embeddings '{a.bge}' -> CPU")
        EMBED = (AutoTokenizer.from_pretrained(a.bge), OVModelForFeatureExtraction.from_pretrained(a.bge))
    except Exception as e:
        print(f"  Embeddings disabled ({e}) -- OCR + LLM still work. Install optimum[openvino] to enable.")

    API_KEY = a.api_key
    if a.host == "0.0.0.0" and not API_KEY:
        print("  WARNING: binding to 0.0.0.0 with NO --api-key — the inference API is open to the whole network.")
    print(f"  auth: {'Bearer token required' if API_KEY else 'none (local-only bind)'}")
    print(f"\nReady -> http://{a.host}:{a.port}  (point the n8n credential's Gateway URL here)")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
