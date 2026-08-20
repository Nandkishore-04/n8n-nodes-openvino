#!/usr/bin/env python3
"""Export the voice-layer models to OpenVINO IR: Whisper (ASR) + SpeechT5 (TTS).

One-time prep, mirrors scripts/convert_clip.py. Uses optimum-cli, installed with
`pip install "optimum[openvino]"`. The base checkpoints download from Hugging Face once,
then everything runs locally on the Intel chips.

  python scripts/convert_speech_models.py --out deployment/models

Produces:
  <out>/whisper-base/    -> gateway --asr-model  (openvino-genai WhisperPipeline; CPU/GPU/NPU)
  <out>/speecht5-tts/    -> gateway --tts-model   (openvino-genai Text2SpeechPipeline; CPU/GPU)

Then start the gateway with the voice flags, e.g.:
  python scripts/native_gateway.py --models deployment/models --ocr-engine vlm ^
    --vlm-model deployment/models/qwen2.5-vl-7b --llm <qwen3-8b> ^
    --ocr-device GPU --llm-device GPU --clip-device NPU ^
    --asr-model deployment/models/whisper-base --asr-device GPU ^
    --tts-model deployment/models/speecht5-tts  --tts-device CPU --port 8000
"""
import argparse
import os
import subprocess
import sys


def run(cmd):
    print("+ " + " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="deployment/models", help="output root for the IR dirs")
    ap.add_argument("--asr-id", default="openai/whisper-base", help="ASR checkpoint (whisper-base is small + fast; -small/-medium for accuracy)")
    ap.add_argument("--tts-id", default="microsoft/speecht5_tts", help="TTS checkpoint")
    ap.add_argument("--vocoder-id", default="microsoft/speecht5_hifigan", help="SpeechT5 vocoder (required)")
    ap.add_argument("--weight-format", default="int8", choices=["fp16", "int8"], help="int8 = smaller/faster; fp16 = max quality")
    ap.add_argument("--skip-asr", action="store_true")
    ap.add_argument("--skip-tts", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    asr_out = os.path.join(a.out, "whisper-base")
    tts_out = os.path.join(a.out, "speecht5-tts")

    if not a.skip_asr:
        print(f"\n=== Whisper ASR -> {asr_out} ===")
        run(["optimum-cli", "export", "openvino", "--model", a.asr_id,
             "--weight-format", a.weight_format, "--trust-remote-code", asr_out])

    if not a.skip_tts:
        print(f"\n=== SpeechT5 TTS (+ vocoder) -> {tts_out} ===")
        # The vocoder must be supplied via --model-kwargs (JSON), per the OpenVINO GenAI
        # speech-generation docs; Text2SpeechPipeline then loads model + vocoder from this dir.
        run(["optimum-cli", "export", "openvino", "--model", a.tts_id,
             "--model-kwargs", '{"vocoder": "%s"}' % a.vocoder_id,
             "--weight-format", a.weight_format, "--trust-remote-code", tts_out])

    print("\nDone. Add these flags to the gateway:")
    if not a.skip_asr:
        print(f"  --asr-model {asr_out} --asr-device GPU")
    if not a.skip_tts:
        print(f"  --tts-model {tts_out} --tts-device CPU")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\nexport failed ({e}).", file=sys.stderr)
        print("Check that `pip install \"optimum[openvino]\"` is done and you have internet for the one-time download.", file=sys.stderr)
        sys.exit(1)
