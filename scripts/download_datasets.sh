#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"
RAW_DIR="${DATA_DIR}/raw"

COCO_TRAIN_URL="http://images.cocodataset.org/zips/train2017.zip"
COCO_VAL_URL="http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
OPENTHERMALPOSE2_URL="https://huggingface.co/datasets/issai/OpenThermalPose2/resolve/main/OpenThermalPose2.zip"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/download_datasets.sh all
  ./scripts/download_datasets.sh coco
  ./scripts/download_datasets.sh openthermalpose2

Environment variables:
  DATA_DIR_OVERRIDE   Override dataset output directory
  CURL_OPTS           Extra options passed to curl

Examples:
  ./scripts/download_datasets.sh all
  DATA_DIR_OVERRIDE=/mnt/datasets ./scripts/download_datasets.sh coco
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

download_file() {
  local url="$1"
  local out="$2"
  if [[ -f "$out" ]]; then
    log "skip download, file already exists: $out"
    return
  fi
  mkdir -p "$(dirname "$out")"
  if command -v curl >/dev/null 2>&1; then
    log "downloading $(basename "$out")"
    # shellcheck disable=SC2086
    curl -L --fail ${CURL_OPTS:-} -o "$out" "$url"
  elif command -v wget >/dev/null 2>&1; then
    log "downloading $(basename "$out")"
    wget -O "$out" "$url"
  else
    fail "curl or wget is required"
  fi
}

extract_zip() {
  local archive="$1"
  local dest="$2"
  mkdir -p "$dest"
  if command -v unzip >/dev/null 2>&1; then
    unzip -qo "$archive" -d "$dest"
  elif command -v bsdtar >/dev/null 2>&1; then
    bsdtar -xf "$archive" -C "$dest"
  elif command -v python3 >/dev/null 2>&1; then
    python3 - <<PY
from pathlib import Path
from zipfile import ZipFile
archive = Path(r"$archive")
dest = Path(r"$dest")
dest.mkdir(parents=True, exist_ok=True)
with ZipFile(archive) as zf:
    zf.extractall(dest)
PY
  else
    fail "unzip, bsdtar, or python3 is required to extract zip files"
  fi
}

ensure_data_dir() {
  if [[ -n "${DATA_DIR_OVERRIDE:-}" ]]; then
    DATA_DIR="${DATA_DIR_OVERRIDE}"
    RAW_DIR="${DATA_DIR}/raw"
  fi
  mkdir -p "$DATA_DIR" "$RAW_DIR"
}

download_coco() {
  ensure_data_dir
  mkdir -p "${DATA_DIR}/coco" "${DATA_DIR}/coco/annotations"
  download_file "$COCO_TRAIN_URL" "${RAW_DIR}/train2017.zip"
  download_file "$COCO_VAL_URL" "${RAW_DIR}/val2017.zip"
  download_file "$COCO_ANN_URL" "${RAW_DIR}/annotations_trainval2017.zip"

  if [[ ! -d "${DATA_DIR}/coco/train2017" ]]; then
    log "extracting COCO train2017"
    extract_zip "${RAW_DIR}/train2017.zip" "${DATA_DIR}/coco"
  fi
  if [[ ! -d "${DATA_DIR}/coco/val2017" ]]; then
    log "extracting COCO val2017"
    extract_zip "${RAW_DIR}/val2017.zip" "${DATA_DIR}/coco"
  fi
  if [[ ! -f "${DATA_DIR}/coco/annotations/person_keypoints_train2017.json" ]]; then
    log "extracting COCO annotations"
    extract_zip "${RAW_DIR}/annotations_trainval2017.zip" "${DATA_DIR}/coco"
  fi
}

normalize_openthermalpose2() {
  local extracted_root="$1"
  local target_root="${DATA_DIR}/openthermalpose2"
  local target_images="${target_root}/images"
  local target_annotations="${target_root}/annotations"
  local image_probe
  local json_probe

  mkdir -p "$target_images" "$target_annotations"

  image_probe="$(find "$extracted_root" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | head -n 1 || true)"
  json_probe="$(find "$extracted_root" -type f -iname '*.json' | head -n 1 || true)"

  if [[ -n "$json_probe" ]]; then
    cp -f "$json_probe" "${target_annotations}/train.json"
    log "annotation json copied to ${target_annotations}/train.json"
  else
    log "no json annotation found automatically; raw files kept in ${extracted_root}"
  fi

  if [[ -n "$image_probe" ]]; then
    local image_dir
    image_dir="$(dirname "$image_probe")"
    if [[ ! -e "${target_images}/.linked_from_raw" ]]; then
      cp -R "$image_dir"/. "$target_images"/
      : > "${target_images}/.linked_from_raw"
    fi
    log "images copied to ${target_images}"
  else
    log "no image directory detected automatically; raw files kept in ${extracted_root}"
  fi

  cat <<EOF

[LiteThermPose] OpenThermalPose2 extraction finished.
[LiteThermPose] Expected training paths:
  images:      ${target_images}
  annotations: ${target_annotations}/train.json

[LiteThermPose] If the dataset uses a non-COCO annotation format, convert it before training.
[LiteThermPose] Raw extracted files are preserved at:
  ${extracted_root}

EOF
}

download_openthermalpose2() {
  ensure_data_dir
  local raw_zip="${RAW_DIR}/OpenThermalPose2.zip"
  local raw_extract="${RAW_DIR}/openthermalpose2_extracted"
  download_file "$OPENTHERMALPOSE2_URL" "$raw_zip"
  if [[ ! -d "$raw_extract" ]]; then
    log "extracting OpenThermalPose2"
    extract_zip "$raw_zip" "$raw_extract"
  fi
  normalize_openthermalpose2 "$raw_extract"
}

main() {
  require_cmd find
  case "${1:-}" in
    all)
      download_coco
      download_openthermalpose2
      ;;
    coco)
      download_coco
      ;;
    openthermalpose2)
      download_openthermalpose2
      ;;
    -h|--help|help|"")
      usage
      ;;
    *)
      usage
      fail "unknown target: $1"
      ;;
  esac
}

main "${1:-}"
