from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

SUBJECT_PREFIX = "S"
KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7],
]
BODY_PARTS = {
    "head": (0, 1, 2, 3, 4),
    "torso": (5, 6, 11, 12),
    "left_arm": (5, 7, 9),
    "right_arm": (6, 8, 10),
    "left_leg": (11, 13, 15),
    "right_leg": (12, 14, 16),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LWIRPOSE to COCO-style JSON for LiteThermPose.")
    parser.add_argument("--source", type=str, required=True, help="Path to LWIRPose root (containing S1..S7)")
    parser.add_argument("--target", type=str, required=True, help="Target root, e.g. data/lwirpose")
    parser.add_argument("--test-subjects", type=str, default="S1", help="Comma-separated subject ids used as test/val")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking")
    return parser.parse_args()


def read_pose_points(label_path: Path) -> list[float]:
    points: list[float] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        sx, sy = line.split()[:2]
        x = float(sx)
        y = float(sy)
        points.extend([x, y, 2.0])
    if len(points) != len(KEYPOINT_NAMES) * 3:
        raise ValueError(f"{label_path} does not contain {len(KEYPOINT_NAMES)} keypoints.")
    return points


def compute_bbox(keypoints: list[float], width: int, height: int) -> list[float]:
    xs = []
    ys = []
    for index in range(0, len(keypoints), 3):
        x, y, vis = keypoints[index : index + 3]
        if vis > 0:
            xs.append(float(x))
            ys.append(float(y))
    if not xs or not ys:
        return [0.0, 0.0, float(width), float(height)]
    x1 = max(min(xs) - 10.0, 0.0)
    y1 = max(min(ys) - 10.0, 0.0)
    x2 = min(max(xs) + 10.0, float(width - 1))
    y2 = min(max(ys) + 10.0, float(height - 1))
    return [x1, y1, max(x2 - x1, 1.0), max(y2 - y1, 1.0)]


def compute_body_parts(keypoints: list[float]) -> dict[str, list[float]]:
    coords = []
    for index in range(0, len(keypoints), 3):
        coords.append((float(keypoints[index]), float(keypoints[index + 1]), float(keypoints[index + 2])))
    parts: dict[str, list[float]] = {}
    for name, indices in BODY_PARTS.items():
        visible = [(coords[idx][0], coords[idx][1]) for idx in indices if coords[idx][2] > 0]
        if not visible:
            continue
        x = sum(item[0] for item in visible) / len(visible)
        y = sum(item[1] for item in visible) / len(visible)
        parts[name] = [x, y]
    return parts


def ensure_link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def collect_records(source_root: Path, test_subjects: set[str], target_root: Path, copy_images: bool) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    images_train: list[dict] = []
    anns_train: list[dict] = []
    images_test: list[dict] = []
    anns_test: list[dict] = []
    image_id = 1
    ann_id = 1

    for subject_dir in sorted(path for path in source_root.iterdir() if path.is_dir() and path.name.startswith(SUBJECT_PREFIX)):
        subject_name = subject_dir.name
        split_name = "test" if subject_name in test_subjects else "train"
        for action_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            images_dir = action_dir / "IR"
            labels_dir = action_dir / "IR_posepoints"
            if not images_dir.is_dir() or not labels_dir.is_dir():
                continue
            for image_path in sorted(images_dir.glob("*.jpg")):
                label_name = f"ir_{image_path.stem}.txt"
                label_path = labels_dir / label_name
                if not label_path.exists():
                    continue
                with Image.open(image_path) as image:
                    width, height = image.size
                target_image = target_root / "images" / split_name / subject_name / action_dir.name / image_path.name
                ensure_link_or_copy(image_path, target_image, copy_images=copy_images)
                file_name = str(target_image.relative_to(target_root / "images")).replace("\\", "/")
                image_entry = {
                    "id": image_id,
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                }
                keypoints = read_pose_points(label_path)
                bbox = compute_bbox(keypoints, width, height)
                ann_entry = {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "iscrowd": 0,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "keypoints": keypoints,
                    "num_keypoints": len(KEYPOINT_NAMES),
                    "body_parts": compute_body_parts(keypoints),
                    "subject": subject_name,
                    "action": action_dir.name,
                }
                if split_name == "train":
                    images_train.append(image_entry)
                    anns_train.append(ann_entry)
                else:
                    images_test.append(image_entry)
                    anns_test.append(ann_entry)
                image_id += 1
                ann_id += 1
    return images_train, anns_train, images_test, anns_test


def write_coco(path: Path, images: list[dict], annotations: list[dict]) -> None:
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "keypoints": KEYPOINT_NAMES,
                "skeleton": SKELETON,
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_root = Path(args.source)
    target_root = Path(args.target)
    test_subjects = {item.strip() for item in args.test_subjects.split(",") if item.strip()}
    images_train, anns_train, images_test, anns_test = collect_records(
        source_root=source_root,
        test_subjects=test_subjects,
        target_root=target_root,
        copy_images=args.copy_images,
    )
    write_coco(target_root / "annotations" / "train.json", images_train, anns_train)
    write_coco(target_root / "annotations" / "val.json", images_test, anns_test)
    write_coco(target_root / "annotations" / "test.json", images_test, anns_test)
    print(f"train images: {len(images_train)}")
    print(f"test images: {len(images_test)}")
    print(f"target root: {target_root}")


if __name__ == "__main__":
    main()
