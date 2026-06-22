"""text-image-super-resolution-0001 wrapper (OVMS-served as 'text-sr').

Open Model Zoo text-image super-resolution. Verified I/O contract against OVMS:
  input  : name '0', shape [1, 1, 360, 640]  (N,C,H,W) grayscale, values in [0, 1]
  output : 'predictions', shape [1, 1, 1080, 1920]  (3x upscale) grayscale

Used to sharpen LOW-QUALITY IMAGES before OCR — the 'enhance' path the agent triggers
via retry_document_extraction. (PDFs use higher DPI instead; this is for raster images
where there's no DPI lever.)

Caveat: the model has a fixed 360x640 input, so a full page is resized to fit. It was
designed for small text-region crops, so it helps most on genuinely low-resolution
images; on already-sharp pages the downscale-to-640 can be a wash. Best-effort —
callers should fall back to the original image on any failure.
"""
import os
import cv2
import numpy as np
import requests

OVMS_URL = os.environ.get("OVMS_URL", "http://localhost:9001")
IN_W, IN_H = 640, 360  # model's fixed input (W, H)


SCALE = 3  # model upscales exactly 3x (360x640 -> 1080x1920)


def super_resolve(bgr, ovms_url=None, timeout=60):
    """Run a BGR image through text-sr; return an enhanced BGR image. Raises on failure.

    The model's input is a fixed 360x640, so we LETTERBOX (resize preserving aspect ratio,
    pad the remainder) instead of squashing — squashing distorts text and destroys OCR.
    After the 3x super-resolution we crop the content region back out (dropping padding).
    """
    ovms_url = ovms_url or OVMS_URL
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Fit within 360x640 preserving aspect; place at top-left, pad right/bottom.
    scale = min(IN_H / h, IN_W / w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((IN_H, IN_W), 255, np.uint8)  # white pad (text is dark on light)
    canvas[:new_h, :new_w] = resized

    canvas01 = canvas.astype(np.float32) / 255.0  # (360, 640) in [0, 1]
    resp = requests.post(
        f"{ovms_url}/v1/models/text-sr:predict",
        json={"instances": [{"0": canvas01[np.newaxis, :, :].tolist()}]},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    out = data.get("predictions")
    if out is None:  # named-output fallback
        out = next(iter(data.values()))

    residual = np.squeeze(np.array(out, dtype=np.float32))  # (1080, 1920)
    # text-sr is a RESIDUAL model: it predicts the high-frequency detail to ADD to a
    # bicubic upscale, not the full image. Reconstruct = bicubic(input) + residual.
    bicubic = cv2.resize(canvas01, (IN_W * SCALE, IN_H * SCALE), interpolation=cv2.INTER_CUBIC)
    recon = np.clip(bicubic + residual, 0.0, 1.0)
    img8 = (recon * 255.0).astype(np.uint8)

    # Crop the content region back out (padding scales by the same 3x factor).
    crop = img8[: new_h * SCALE, : new_w * SCALE]
    return cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
