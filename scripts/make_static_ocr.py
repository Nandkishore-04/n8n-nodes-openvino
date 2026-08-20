#!/usr/bin/env python3
"""Re-export PP-OCR det + rec with STATIC input shapes so they compile on the NPU.

The Intel NPU rejects dynamic (variable-size) inputs -- PP-OCR's detector is dynamic,
which is why `--ocr-device NPU` crashes. This reshapes:
  - detector   -> a fixed square  [1, 3, S, S]   (default S=960)
  - recognizer -> its padded size [1, 3, 48, 320]
and saves them as new model dirs alongside the originals.

Usage (PowerShell):
  python scripts\\make_static_ocr.py --models deployment\\models --det-size 960

Creates:
  <models>\\ppocr-det-static\\1\\model.xml(.bin)
  <models>\\ppocr-rec-static\\1\\model.xml(.bin)

Then run the gateway with:  --static-ocr --ocr-device NPU
"""
import argparse
import os

import openvino as ov


def reshape_save(core, src_dir, dst_dir, shape):
    src = os.path.join(src_dir, "1", "model.xml")
    m = core.read_model(src)
    m.reshape(shape)  # single input -> positional static shape
    out_dir = os.path.join(dst_dir, "1")
    os.makedirs(out_dir, exist_ok=True)
    ov.save_model(m, os.path.join(out_dir, "model.xml"))
    print(f"  {os.path.basename(dst_dir)} <- static {shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="dir containing ppocr-det / ppocr-rec")
    ap.add_argument("--det-size", type=int, default=960, help="fixed square size for the detector")
    ap.add_argument("--rec-h", type=int, default=48)
    ap.add_argument("--rec-w", type=int, default=320)
    a = ap.parse_args()

    core = ov.Core()
    s = a.det_size
    print("Re-exporting OCR models with static shapes:")
    reshape_save(core, os.path.join(a.models, "ppocr-det"),
                 os.path.join(a.models, "ppocr-det-static"), [1, 3, s, s])
    reshape_save(core, os.path.join(a.models, "ppocr-rec"),
                 os.path.join(a.models, "ppocr-rec-static"), [1, 3, a.rec_h, a.rec_w])
    print("Done. Run the gateway with:  --static-ocr --ocr-device NPU")


if __name__ == "__main__":
    main()
