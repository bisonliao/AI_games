#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 RUN_NAME [afterBC.train arguments...]" >&2
  exit 2
fi

run_name=$1
shift
exec conda run -n mygames python -m afterBC.train --run-name "$run_name" "$@"
