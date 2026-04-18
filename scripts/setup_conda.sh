#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/environment.yml"
ENV_NAME="${CONDA_ENV_NAME:-lite-therm-pose}"
CPU_ONLY="${CPU_ONLY:-0}"
CUDA_VERSION="${CUDA_VERSION:-12.1}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/setup_conda.sh

Optional environment variables:
  CONDA_ENV_NAME   Override the conda environment name
  CPU_ONLY         Set to 1 to create a CPU-only environment instead of CUDA
  CUDA_VERSION     CUDA runtime version for PyTorch packages (default: 12.1)

Examples:
  ./scripts/setup_conda.sh
  CONDA_ENV_NAME=thermal-pose ./scripts/setup_conda.sh
  CPU_ONLY=1 ./scripts/setup_conda.sh
  CUDA_VERSION=12.1 ./scripts/setup_conda.sh
EOF
}

log() {
  printf '[LiteThermPose] %s\n' "$*"
}

fail() {
  printf '[LiteThermPose][ERROR] %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
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

  fail "conda が見つかりません。Ubuntu なら Anaconda/Miniconda を入れ、PATH を通すか ~/miniconda3/bin/conda のような一般的な場所に配置してください。"
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
  CONDA_EXE="$(find_conda)"

  log "creating or updating conda environment: ${ENV_NAME}"
  "$CONDA_EXE" env remove -n "$ENV_NAME" -y >/dev/null 2>&1 || true
  "$CONDA_EXE" env create -n "$ENV_NAME" -f "$ENV_FILE"

  if [[ "$CPU_ONLY" == "1" ]]; then
    "$CONDA_EXE" install -n "$ENV_NAME" -y \
      "pytorch>=2.2" \
      "torchvision>=0.17" \
      "cpuonly" \
      -c pytorch -c conda-forge
  else
    "$CONDA_EXE" install -n "$ENV_NAME" -y \
      "pytorch>=2.2" \
      "torchvision>=0.17" \
      "pytorch-cuda=${CUDA_VERSION}" \
      -c pytorch -c nvidia
  fi

  "$CONDA_EXE" run -n "$ENV_NAME" pip install -e "$ROOT_DIR"

  log "verifying torch CUDA availability"
  "$CONDA_EXE" run -n "$ENV_NAME" python "${ROOT_DIR}/scripts/verify_gpu.py" || true

  log "activating environment instructions:"
  printf '  source "$(dirname "%s")/../etc/profile.d/conda.sh"\n' "$CONDA_EXE"
  printf '  conda activate %s\n' "$ENV_NAME"
  printf '  python scripts/verify_gpu.py\n'
  printf '  python train_detector.py --config configs/coco_pretrain.yaml --output outputs/coco_detector\n'
}

main "${1:-}"
