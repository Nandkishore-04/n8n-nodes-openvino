#!/usr/bin/env python3
"""Convert BSRGAN (4x super-resolution) to an OpenVINO IR for the gateway's enhance path.

BSRGAN sharpens low-quality scans before OCR -- the lever the agent pulls via
retry_document_extraction / the "Re-OCR (Enhanced)" node. The native gateway loads the IR
this script writes to <models>/bsrgan/1/model.xml and runs it on GPU/CPU (not NPU: it's a
heavy GAN with dynamic shapes).

Pipeline:  BSRGAN.pth (RRDBNet weights)  ->  ONNX (dynamic H/W)  ->  OpenVINO IR

Usage (needs torch; run once, offline -- not on the AI PC at demo time):
  pip install torch onnx openvino numpy
  # get the weights first (BSRGAN model zoo): https://github.com/cszn/BSRGAN
  python scripts/convert_bsrgan.py --weights BSRGAN.pth
  # or let it try to download:
  python scripts/convert_bsrgan.py

Output: deployment/models/bsrgan/1/model.xml (+ .bin)
"""
import argparse
import os
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default weights mirror (BSRGAN / KAIR). If this 404s, download BSRGAN.pth manually from
# https://github.com/cszn/BSRGAN  and pass it with --weights.
DEFAULT_URL = "https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN.pth"


# --- RRDBNet (the BSRGAN generator). Module names match the official BSRGAN.pth state dict,
#     so load_state_dict(strict=True) succeeds. ---------------------------------------------
class ResidualDenseBlock5C(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.RDB1 = ResidualDenseBlock5C(nf, gc)
        self.RDB2 = ResidualDenseBlock5C(nf, gc)
        self.RDB3 = ResidualDenseBlock5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4):
        super().__init__()
        self.sf = sf
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.RRDB_trunk = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode="nearest")))
        if self.sf == 4:
            fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.HRconv(fea)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="", help="path to BSRGAN.pth (downloads if omitted)")
    ap.add_argument("--url", default=DEFAULT_URL, help="weights download URL if --weights absent")
    ap.add_argument("--out", default=os.path.join("deployment", "models", "bsrgan", "1"),
                    help="output dir for model.xml/.bin")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()

    weights = a.weights
    if not weights:
        weights = os.path.join(a.out, "BSRGAN.pth")
        os.makedirs(a.out, exist_ok=True)
        if not os.path.exists(weights):
            print(f"Downloading weights from {a.url} ...")
            try:
                urllib.request.urlretrieve(a.url, weights)
            except Exception as e:
                raise SystemExit(
                    f"Download failed ({e}).\nDownload BSRGAN.pth manually from "
                    f"https://github.com/cszn/BSRGAN and re-run with --weights <path>.")

    print(f"Loading {weights} into RRDBNet ...")
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)
    sd = torch.load(weights, map_location="cpu")
    sd = sd.get("params_ema") or sd.get("params") or sd  # tolerate basicsr-style wrappers
    model.load_state_dict(sd, strict=True)
    model.eval()

    os.makedirs(a.out, exist_ok=True)
    onnx_path = os.path.join(a.out, "bsrgan.onnx")
    dummy = torch.randn(1, 3, 128, 128)
    print("Exporting ONNX (dynamic H/W) ...")
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["output"], opset_version=a.opset,
        dynamic_axes={"input": {0: "b", 2: "h", 3: "w"}, "output": {0: "b", 2: "h4", 3: "w4"}},
    )

    print("Converting ONNX -> OpenVINO IR ...")
    import openvino as ov
    ov_model = ov.convert_model(onnx_path)
    ov.save_model(ov_model, os.path.join(a.out, "model.xml"))
    os.remove(onnx_path)
    print(f"Done -> {os.path.join(a.out, 'model.xml')}")
    print("The native gateway auto-loads this; enable per request with enhance=true.")


if __name__ == "__main__":
    main()
