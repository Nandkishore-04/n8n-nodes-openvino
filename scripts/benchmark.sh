#!/usr/bin/env bash
# Device benchmark for OpenVINO IR models — latency + throughput per device.
# Runs locally (CPU/GPU) and on Intel Tiber Cloud (adds NPU). Wraps OpenVINO's benchmark_app.
#
# Usage:
#   scripts/benchmark.sh <model.xml> ["shape"] ["devices"]
# Examples:
#   scripts/benchmark.sh deployment/models/text-classifier/1/model.xml "input_ids[1,128],attention_mask[1,128]"
#   scripts/benchmark.sh deployment/models/ppocr-det/1/model.xml "x[1,3,640,640]" "CPU GPU NPU"
set -uo pipefail

MODEL="${1:?usage: benchmark.sh <model.xml> [shape] [devices]}"
SHAPE="${2:-}"
DEVICES="${3:-CPU GPU NPU}"
DURATION="${BENCH_SECONDS:-10}"

command -v benchmark_app >/dev/null 2>&1 || { echo "benchmark_app not found — pip install openvino"; exit 1; }

shape_arg=()
[ -n "$SHAPE" ] && shape_arg=(-shape "$SHAPE")

echo "Model:    $MODEL"
echo "Shape:    ${SHAPE:-<static>}"
echo "Duration: ${DURATION}s per run"
echo
printf "%-8s | %-8s | %-14s | %-12s\n" "Device" "Hint" "Latency (med)" "Throughput"
printf -- "---------+----------+----------------+-------------\n"

for dev in $DEVICES; do
	for hint in latency throughput; do
		out="$(benchmark_app -m "$MODEL" -d "$dev" -hint "$hint" -t "$DURATION" "${shape_arg[@]}" 2>&1)"
		if echo "$out" | grep -qiE "Throughput:"; then
			lat="$(echo "$out" | grep -iE "Median:" | grep -oE "[0-9.]+ ms" | head -1)"
			thr="$(echo "$out" | grep -iE "Throughput:" | grep -oE "[0-9.]+ FPS" | head -1)"
			printf "%-8s | %-8s | %-14s | %-12s\n" "$dev" "$hint" "${lat:-n/a}" "${thr:-n/a}"
		else
			printf "%-8s | %-8s | %-14s | %-12s\n" "$dev" "$hint" "unavailable" "unavailable"
		fi
	done
done
