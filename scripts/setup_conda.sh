#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/environment.yml"
ENV_NAME="${CONDA_ENV_NAME:-lite-therm-pose}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/setup_conda.sh

Optional environment variables:
  CONDA_ENV_NAME   Override the conda environment name

Examples:
  ./scripts/setup_conda.sh
  CONDA_ENV_NAME=thermal-pose ./scripts/setup_conda.sh
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

  require_cmd conda
  [[ -f "$ENV_FILE" ]] || fail "environment file not found: $ENV_FILE"

  log "creating or updating conda environment: ${ENV_NAME}"
  conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1 || true
  conda env create -n "$ENV_NAME" -f "$ENV_FILE"

  log "activating environment instructions:"
  printf '  conda activate %s\n' "$ENV_NAME"
  printf '  python train_detector.py --config configs/coco_pretrain.yaml --output outputs/coco_detector\n'
}

main "${1:-}"
