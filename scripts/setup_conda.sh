#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/environment.yml"
ENV_NAME="${CONDA_ENV_NAME:-lite-therm-pose}"
CPU_ONLY="${CPU_ONLY:-0}"
CUDA_VERSION="${CUDA_VERSION:-12.4}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.5.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.0}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/setup_conda.sh

Optional environment variables:
  CONDA_ENV_NAME       Override the conda environment name
  CPU_ONLY             Set to 1 to create a CPU-only environment instead of CUDA
  CUDA_VERSION         CUDA runtime version for PyTorch packages (default: 12.4)
  PYTORCH_VERSION      PyTorch version to install (default: 2.5.0)
  TORCHVISION_VERSION  Torchvision version to install (default: 0.20.0)

Examples:
  ./scripts/setup_conda.sh
  CONDA_ENV_NAME=thermal-pose ./scripts/setup_conda.sh
  CPU_ONLY=1 ./scripts/setup_conda.sh
  CUDA_VERSION=12.4 ./scripts/setup_conda.sh
EOF
}

log() {
  printf '[LiteThermPose] %s\n' "$*"
}

fail() {
  printf '[LiteThermPose][ERROR] %s\n' "$*" >&2
  exit 1
}

find_conda() {
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return
  fi

  local candidates=(
    "${HOME}/anaconda3/bin/conda"
    "${HOME}/miniconda3/bin/conda"
    "/opt/conda/bin/conda"
    "/usr/local/anaconda3/bin/conda"
    "/usr/local/miniconda3/bin/conda"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  fail "conda was not found. Install Anaconda or Miniconda and make sure conda is on PATH."
}

main() {
  case "${1:-}" in
    -h|--help|help)
      usage
      exit 0
      ;;
    "")
      ;;
    *)
      usage
      fail "unknown argument: $1"
      ;;
  esac

  [[ -f "$ENV_FILE" ]] || fail "environment file not found: $ENV_FILE"
  local conda_exe
  conda_exe="$(find_conda)"

  log "creating or updating conda environment: ${ENV_NAME}"
  "$conda_exe" env remove -n "$ENV_NAME" -y >/dev/null 2>&1 || true
  "$conda_exe" env create -n "$ENV_NAME" -f "$ENV_FILE"

  log "removing conflicting pip torch packages if present"
  "$conda_exe" run -n "$ENV_NAME" python -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true

  if [[ "$CPU_ONLY" == "1" ]]; then
    "$conda_exe" install -n "$ENV_NAME" -y \
      "pytorch=${PYTORCH_VERSION}" \
      "torchvision=${TORCHVISION_VERSION}" \
      "cpuonly" \
      -c pytorch -c conda-forge
  else
    "$conda_exe" install -n "$ENV_NAME" -y \
      "pytorch=${PYTORCH_VERSION}" \
      "torchvision=${TORCHVISION_VERSION}" \
      "pytorch-cuda=${CUDA_VERSION}" \
      -c pytorch -c nvidia
  fi

  "$conda_exe" run -n "$ENV_NAME" python -m pip install --no-deps -e "$ROOT_DIR"

  log "verifying torch CUDA availability"
  "$conda_exe" run -n "$ENV_NAME" python "${ROOT_DIR}/scripts/verify_gpu.py" || true

  log "activation instructions:"
  printf '  source "$(dirname "%s")/../etc/profile.d/conda.sh"\n' "$conda_exe"
  printf '  conda activate %s\n' "$ENV_NAME"
  printf '  python scripts/verify_gpu.py\n'
  printf '  python train_detector.py --config configs/coco_pretrain.yaml --output outputs/coco_detector\n'
}

main "${1:-}"
