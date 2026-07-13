#!/usr/bin/env bash
set -Eeuo pipefail

# Temporary bootstrap script for the Gomoku RL development environment.
# PyTorch's cu128 wheels include the CUDA user-space runtime and cuDNN; only a
# compatible NVIDIA driver is required on the host.

ENV_NAME="mygames"
PYTHON_VERSION="3.11"
PYTORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu128"
export PYGAME_HIDE_SUPPORT_PROMPT=1

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda was not found in PATH." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Conda environment '${ENV_NAME}' already exists; reusing it."
else
  echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
  conda create --name "${ENV_NAME}" --yes "python=${PYTHON_VERSION}" pip
fi

conda run --name "${ENV_NAME}" python -m pip install --upgrade pip

echo "Checking PyTorch with CUDA support..."
if conda run --name "${ENV_NAME}" python -c 'import torch, torchvision, torchaudio; raise SystemExit(0 if torch.version.cuda is not None else 1)' >/dev/null 2>&1; then
  echo "CUDA-enabled PyTorch is already installed; skipping PyTorch download."
else
  echo "Installing PyTorch with its CUDA 12.8 runtime and cuDNN..."
  conda run --name "${ENV_NAME}" python -m pip install \
    torch torchvision torchaudio \
    --index-url "${PYTORCH_CUDA_INDEX}"
fi

echo "Installing common RL/development utilities..."
MISSING_PACKAGES="$(conda run --name "${ENV_NAME}" python -c 'import importlib.util
packages = {
    "numpy": "numpy",
    "gymnasium": "gymnasium",
    "pygame": "pygame",
    "tensorboard": "tensorboard",
    "tqdm": "tqdm",
    "pytest": "pytest",
}
print(" ".join(pkg for pkg, module in packages.items() if importlib.util.find_spec(module) is None))
')"
if [[ -n "${MISSING_PACKAGES}" ]]; then
  conda run --name "${ENV_NAME}" python -m pip install ${MISSING_PACKAGES}
else
  echo "Common RL/development utilities are already installed."
fi

echo "Verifying the installation..."
conda run --no-capture-output --name "${ENV_NAME}" python -c '
import sys

import torch
import gymnasium
import pygame

print(f"Python:        {sys.version.split()[0]}")
print(f"PyTorch:       {torch.__version__}")
print(f"CUDA runtime:  {torch.version.cuda}")
print(f"cuDNN:         {torch.backends.cudnn.version()}")
print(f"Gymnasium:     {gymnasium.__version__}")
print(f"Pygame:        {pygame.version.ver}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise SystemExit(
        "PyTorch was installed, but it cannot access the GPU. "
        "Check the NVIDIA driver and whether this shell/container exposes the GPU."
    )

device = torch.device("cuda")
x = torch.rand((1024, 1024), device=device)
y = x @ x
torch.cuda.synchronize()
print(f"GPU:           {torch.cuda.get_device_name(0)}")
print(f"GPU smoke test: OK ({y.shape[0]}x{y.shape[1]} matrix multiplication)")
'

echo
echo "Environment '${ENV_NAME}' is ready. Activate it with:"
echo "  conda activate ${ENV_NAME}"
