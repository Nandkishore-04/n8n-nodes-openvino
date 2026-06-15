"""PP-OCRv5 two-stage OCR: detection (DB) -> crop -> recognition (CTC).

Ported verbatim from the validated pipeline (gsoc-validation/ocr_test.py) — do not
re-derive the pre/post; this is the proven logic. Only adapted for the gateway:
char dict loaded from the shipped .txt, OVMS URL configurable, returns structured JSON.
"""
import os

import cv2
import numpy as np
import requests

OVMS_URL = os.environ.get("OVMS_URL", "http://ovms:9001")
CHAR_DICT_PATH = os.environ.get("PPOCR_DICT_PATH", "/models/ppocr-v5-en-dict.txt")
DET_MODEL = os.environ.get("PPOCR_DET_MODEL", "ppocr-det")
REC_MODEL = os.environ.get("PPOCR_REC_MODEL", "ppocr-rec")

# CTC vocab: idx 0 = blank, 1..N = dict chars, last = space (matches PP-OCRv5).
with open(CHAR_DICT_PATH, encoding="utf-8") as f:
    CHAR_DICT = [line.rstrip("\n") for line in f]
CTC_CHARS = ["[blank]"] + CHAR_DICT + [" "]


def _ovms_infer(model: str, tensor: np.ndarray) -> np.ndarray:
    """Call OVMS KServe v2 infer endpoint with a single FP32 input named 'x'."""
    payload = {
        "inputs": [{
            "name": "x",
            "shape": list(tensor.shape),
            "datatype": "FP32",
            "data": tensor.flatten().tolist(),
        }]
    }
    r = requests.post(f"{OVMS_URL}/v2/models/{model}/infer", json=payload, timeout=60)
    r.raise_for_status()
    out = r.json()["outputs"][0]
    return np.array(out["data"], dtype=np.float32).reshape(out["shape"])


def _preprocess_det(img: np.ndarray, limit: int = 960):
    h, w = img.shape[:2]
    scale = limit / max(h, w)
    nh, nw = int(round(h * scale / 32) * 32), int(round(w * scale / 32) * 32)
    resized = cv2.resize(img, (nw, nh))
    x = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    x = (x - mean) / std
    x = x.transpose(2, 0, 1)[None]  # NCHW
    return x.astype(np.float32), (h / nh, w / nw)


def _db_boxes(prob_map: np.ndarray, ratio, thresh: float = 0.3, box_thresh: float = 0.5):
    """DB postprocess: probability map -> bounding boxes (sorted top-to-bottom)."""
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
        boxes.append((
            max(0, int(x * rw) - pad), max(0, int(y * rh) - pad),
            int((x + w) * rw) + pad, int((y + h) * rh) + pad, conf,
        ))
    boxes.sort(key=lambda b: b[1])
    return boxes


def _preprocess_rec(crop: np.ndarray, h: int = 48, max_w: int = 320) -> np.ndarray:
    ch, cw = crop.shape[:2]
    nw = min(max_w, max(1, int(cw * h / ch)))
    resized = cv2.resize(crop, (nw, h))
    x = resized.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = x.transpose(2, 0, 1)
    padded = np.zeros((3, h, max_w), np.float32)
    padded[:, :, :nw] = x
    return padded[None].astype(np.float32)


def _ctc_decode(logits: np.ndarray) -> str:
    """logits [1, T, V] -> text (collapse repeats, drop blank index 0)."""
    seq = logits[0].argmax(axis=1)
    text, prev = [], -1
    for idx in seq:
        if idx != 0 and idx != prev and idx < len(CTC_CHARS):
            text.append(CTC_CHARS[idx])
        prev = idx
    return "".join(text)


def run_ocr(img: np.ndarray) -> dict:
    """Full OCR on a BGR image -> {text, boxes, confidence}.

    boxes: [{text, bbox: [x1,y1,x2,y2], confidence}], confidence: area-weighted mean.
    """
    x, ratio = _preprocess_det(img)
    det_out = _ovms_infer(DET_MODEL, x)
    prob = det_out[0, 0]
    boxes = _db_boxes(prob, ratio)

    regions, lines = [], []
    total_area = 0.0
    weighted_conf = 0.0
    for (x1, y1, x2, y2, det_conf) in boxes:
        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            continue
        rec_out = _ovms_infer(REC_MODEL, _preprocess_rec(crop))
        text = _ctc_decode(rec_out)
        if not text.strip():
            continue
        area = float((x2 - x1) * (y2 - y1))
        total_area += area
        weighted_conf += det_conf * area
        regions.append({"text": text, "bbox": [x1, y1, x2, y2], "confidence": round(det_conf, 4)})
        lines.append(text)

    confidence = round(weighted_conf / total_area, 4) if total_area else 0.0
    return {"text": "\n".join(lines), "boxes": regions, "confidence": confidence}
