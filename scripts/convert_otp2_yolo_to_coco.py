from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

NUM_KEYPOINTS = 17


def find_image(images_dir: Path, stem: str) -> Path:
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"image not found for label stem '{stem}' in {images_dir}")


def parse_line(line: str, width: int, height: int) -> dict:
    values = [float(item) for item in line.strip().split()]
    expected = 1 + 4 + NUM_KEYPOINTS * 3
    if len(values) != expected:
        raise ValueError(f"expected {expected} values, got {len(values)}")
    _, cx, cy, bw, bh, *rest = values
    x = (cx - bw / 2.0) * width
    y = (cy - bh / 2.0) * height
    bbox_w = bw * width
    bbox_h = bh * height
    keypoints: list[float] = []
    num_keypoints = 0
    for index in range(0, len(rest), 3):
        kx, ky, vis = rest[index : index + 3]
        if vis > 0:
            keypoints.extend([kx * width, ky * height, vis])
            num_keypoints += 1
        else:
            keypoints.extend([0.0, 0.0, 0.0])
    return {
        "bbox": [x, y, bbox_w, bbox_h],
        "area": bbox_w * bbox_h,
        "keypoints": keypoints,
        "num_keypoints": num_keypoints,
    }


def convert_split(split_name: str, split_root: Path, target_images_root: Path, annotation_path: Path) -> None:
    images_dir = split_root / "images"
    labels_dir = split_root / "labels"
    target_split_root = target_images_root / split_name
    target_split_root.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    ann_id = 1

    for image_id, label_path in enumerate(sorted(labels_dir.glob("*.txt")), start=1):
        stem = label_path.stem
        image_path = find_image(images_dir, stem)
        target_image_path = target_split_root / image_path.name
        if not target_image_path.exists():
            shutil.copy2(image_path, target_image_path)
        with Image.open(image_path) as image:
            width, height = image.size
        images.append(
            {
                "id": image_id,
                "file_name": f"{split_name}/{target_image_path.name}",
                "width": width,
                "height": height,
            }
        )
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parsed = parse_line(line, width, height)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "iscrowd": 0,
                    **parsed,
                }
            )
            ann_id += 1

    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "keypoints": [
                    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                    "left_wrist", "right_wrist", "left_hip", "right_hip",
                    "left_knee", "right_knee", "left_ankle", "right_ankle",
                ],
                "skeleton": [
                    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
                    [2, 4], [3, 5], [4, 6], [5, 7],
                ],
            }
        ],
    }
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert OpenThermalPose2 YOLO-pose labels to COCO JSON.")
    parser.add_argument("--source", type=str, required=True, help="Path to otp2_dataset root")
    parser.add_argument("--target", type=str, required=True, help="Path to data/openthermalpose2 root")
    args = parser.parse_args()

    source_root = Path(args.source)
    target_root = Path(args.target)
    target_images_root = target_root / "images"
    target_annotations_root = target_root / "annotations"

    convert_split("train", source_root / "train", target_images_root, target_annotations_root / "train.json")
    convert_split("val", source_root / "val", target_images_root, target_annotations_root / "val.json")


if __name__ == "__main__":
    main()
