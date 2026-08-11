#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_NAME="${1:-bc-9x9-$(date +%Y%m%d-%H%M%S)}"
exec python BC/pipeline.py --run-name "$RUN_NAME"
