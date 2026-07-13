#!/usr/bin/env bash
set -Eeuo pipefail

# Bootstrap mygames by reusing the existing pymarl2 environment.
# Conda cloning is the reliable no-download path: it reuses local package files
# and keeps the Python/PyTorch/CUDA ABI combination intact.

SOURCE_ENV="${SOURCE_ENV:-pymarl2}"
TARGET_ENV="${TARGET_ENV:-mygames}"
RECREATE="${RECREATE:-0}"
STRICT_GPU="${STRICT_GPU:-0}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda was not found in PATH." >&2
  exit 1
fi

env_exists() {
  local env_name="$1"
  conda env list | awk '{print $1}' | grep -Fxq "${env_name}"
}

if ! env_exists "${SOURCE_ENV}"; then
  echo "Error: source conda environment '${SOURCE_ENV}' does not exist." >&2
  exit 1
fi

if env_exists "${TARGET_ENV}"; then
  if [[ "${RECREATE}" == "1" ]]; then
    echo "Removing existing conda environment '${TARGET_ENV}'..."
    conda env remove --name "${TARGET_ENV}" --yes
  else
    echo "Conda environment '${TARGET_ENV}' already exists; leaving it untouched."
    echo "To replace it with a clone of '${SOURCE_ENV}', run:"
    echo "  RECREATE=1 ./${0##*/}"
    echo
    echo "Verifying the existing '${TARGET_ENV}' environment..."
    export STRICT_GPU
    conda run --no-capture-output --name "${TARGET_ENV}" python - <<'PY'
import os
import sys

import torch

print(f"Python:          {sys.version.split()[0]}")
print(f"PyTorch:         {torch.__version__}")
print(f"CUDA runtime:    {torch.version.cuda}")
print(f"cuDNN:           {torch.backends.cudnn.version()}")
print(f"CUDA available:  {torch.cuda.is_available()}")

if torch.cuda.is_available():
    x = torch.rand((1024, 1024), device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"GPU smoke test:  OK ({y.shape[0]}x{y.shape[1]} matrix multiplication)")
elif os.environ.get("STRICT_GPU") == "1":
    raise SystemExit("GPU is not visible to PyTorch.")
else:
    print("GPU smoke test:  skipped; GPU is not visible to PyTorch in this shell.")
PY
    exit 0
  fi
fi

echo "Cloning '${SOURCE_ENV}' into '${TARGET_ENV}' without downloading packages..."
conda create --name "${TARGET_ENV}" --clone "${SOURCE_ENV}" --yes

echo
echo "Verifying '${TARGET_ENV}'..."
export STRICT_GPU
conda run --no-capture-output --name "${TARGET_ENV}" python - <<'PY'
import os
import sys

import torch

print(f"Python:          {sys.version.split()[0]}")
print(f"PyTorch:         {torch.__version__}")
print(f"CUDA runtime:    {torch.version.cuda}")
print(f"cuDNN:           {torch.backends.cudnn.version()}")
print(f"CUDA available:  {torch.cuda.is_available()}")

if torch.cuda.is_available():
    x = torch.rand((1024, 1024), device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"GPU smoke test:  OK ({y.shape[0]}x{y.shape[1]} matrix multiplication)")
elif os.environ.get("STRICT_GPU") == "1":
    raise SystemExit("GPU is not visible to PyTorch.")
else:
    print("GPU smoke test:  skipped; GPU is not visible to PyTorch in this shell.")
PY

echo
echo "Environment '${TARGET_ENV}' is ready. Activate it with:"
echo "  conda activate ${TARGET_ENV}"
