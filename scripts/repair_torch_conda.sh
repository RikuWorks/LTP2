#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${CONDA_ENV_NAME:-lite-therm-pose}"
CUDA_VERSION="${CUDA_VERSION:-12.4}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.5.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.0}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[LiteThermPose][ERROR] conda not found on PATH" >&2
  exit 1
fi

conda run -n "$ENV_NAME" python -m pip uninstall -y torch torchvision torchaudio || true
conda remove -n "$ENV_NAME" -y pytorch torchvision torchaudio pytorch-cuda cpuonly pytorch-mutex || true
conda install -n "$ENV_NAME" -y \
  "pytorch=${PYTORCH_VERSION}" \
  "torchvision=${TORCHVISION_VERSION}" \
  "pytorch-cuda=${CUDA_VERSION}" \
  -c pytorch -c nvidia
conda run -n "$ENV_NAME" python -m pip install --no-deps -e .
conda run -n "$ENV_NAME" python scripts/verify_gpu.py
