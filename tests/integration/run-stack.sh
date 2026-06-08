#!/usr/bin/env bash
# Boot the Podman stack, wait for OVMS + gateway to be ready, then run integration tests.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/deployment"

echo "==> Starting stack..."
podman-compose up -d

echo "==> Waiting for OVMS REST (:9001)..."
for i in $(seq 1 30); do
	if curl -sf http://localhost:9001/v1/config >/dev/null 2>&1; then
		echo "    OVMS ready."
		break
	fi
	sleep 2
	[ "$i" = "30" ] && { echo "    OVMS did not become ready in time"; exit 1; }
done

echo "==> Waiting for gateway (:8000)..."
for i in $(seq 1 30); do
	if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
		echo "    Gateway ready."
		break
	fi
	sleep 2
	[ "$i" = "30" ] && { echo "    Gateway did not become ready in time"; exit 1; }
done

echo "==> Running integration tests..."
cd "$ROOT"
npm run test:integration
