#!/usr/bin/env python3
"""Convert CLIP to an OpenVINO IR for the gateway's NPU document-triage classifier.

The gateway's "glance" (is this a processable document, or a photo/selfie/junk?) runs as a
CLIP zero-shot classifier on the Intel NPU -- a real ViT on the NPU (the deliverable), a
deterministic decision (no prompt oscillation), and a genuine confidence score. This script,
run ONCE offline, exports the CLIP image encoder to a static 224x224 IR (perfect NPU fit) and
precomputes the normalized text embeddings for the doc / non-doc label sets, so at runtime the
gateway only does: image -> NPU image embedding -> cosine-sim vs the cached label embeddings.

Pipeline:  CLIP (HF)  ->  image encoder -> OpenVINO IR (static 1x3x224x224)  +  label text embeddings (.npz)

Usage (needs torch + transformers; run once, offline -- not on the AI PC at demo time):
  pip install torch transformers openvino numpy
  python scripts/convert_clip.py                       # default ViT-B/32
  python scripts/convert_clip.py --model openai/clip-vit-base-patch16

Output:
  deployment/models/clip/1/model.xml (+ .bin)   -- the image encoder (compile on NPU)
  deployment/models/clip/labels.npz             -- text_emb [L,D], is_doc [L], labels [L]
"""
import argparse
import os

import numpy as np
import openvino as ov
import torch
from transformers import CLIPModel, CLIPTokenizer  # tokenizer only — no image processor / PIL needed

# Zero-shot label sets. is_document is decided by the total probability mass on the doc labels,
# so more (varied) labels on each side make the boundary sharper. Tune these to your domain.
DOC_LABELS = [
    "a scanned document", "a page of printed text", "a receipt", "an invoice", "a tax form",
    "a resume", "a business letter", "a book or dictionary page", "a handwritten note",
    "a screenshot of a document", "a table of data",
]
NONDOC_LABELS = [
    "a photo of a person", "a selfie", "a group of people", "an athlete in a sports jersey",
    "a landscape photograph", "a photo of an animal", "a product photograph", "a company logo",
    "a screenshot of a video game", "a meme", "a blank or blurry image",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-base-patch32", help="HF CLIP model id")
    ap.add_argument("--out", default=os.path.join("deployment", "models", "clip"),
                    help="output dir (writes 1/model.xml + labels.npz)")
    a = ap.parse_args()

    print(f"Loading {a.model} ...")
    model = CLIPModel.from_pretrained(a.model)
    tokenizer = CLIPTokenizer.from_pretrained(a.model)
    model.eval()

    # 1) precompute normalized text embeddings for every label (doc + non-doc)
    labels = DOC_LABELS + NONDOC_LABELS
    is_doc = np.array([1] * len(DOC_LABELS) + [0] * len(NONDOC_LABELS), dtype=np.int64)
    with torch.no_grad():
        tok = tokenizer(labels, return_tensors="pt", padding=True)
        pooled = model.text_model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"]).pooler_output
        tfeat = model.text_projection(pooled)              # explicit: pooled text -> shared CLIP space
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
    text_emb = tfeat.cpu().numpy().astype(np.float32)
    logit_scale = float(model.logit_scale.exp().item())
    print(f"  {len(labels)} labels ({len(DOC_LABELS)} doc / {len(NONDOC_LABELS)} non-doc), "
          f"embedding dim {text_emb.shape[1]}, logit_scale {logit_scale:.2f}")

    # 2) export the image encoder (vision tower + projection + L2-normalize) as a static IR.
    #    Static 1x3x224x224 input is exactly what the NPU wants (no dynamic shapes).
    class ImageEncoder(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            pooled = self.m.vision_model(pixel_values=pixel_values).pooler_output
            f = self.m.visual_projection(pooled)           # explicit: pooled image -> shared CLIP space
            return f / f.norm(dim=-1, keepdim=True)

    enc = ImageEncoder(model)
    example = torch.randn(1, 3, 224, 224)
    print("Converting image encoder to OpenVINO IR (static 1x3x224x224) ...")
    ov_model = ov.convert_model(enc, example_input=example)
    ov_model.reshape([1, 3, 224, 224])  # force static shape — the NPU requires it

    xml_dir = os.path.join(a.out, "1")
    os.makedirs(xml_dir, exist_ok=True)
    ov.save_model(ov_model, os.path.join(xml_dir, "model.xml"))
    np.savez(os.path.join(a.out, "labels.npz"),
             text_emb=text_emb, is_doc=is_doc, labels=np.array(labels),
             logit_scale=np.array(logit_scale, dtype=np.float32))
    print(f"Wrote {os.path.join(xml_dir, 'model.xml')} and {os.path.join(a.out, 'labels.npz')}")
    print("Load in the gateway with --clip-model <this dir> --clip-device NPU")


if __name__ == "__main__":
    main()
