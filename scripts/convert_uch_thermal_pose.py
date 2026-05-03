from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize UCH-Thermal-Pose into LiteThermPose COCO-style layout.")
    parser.add_argument("--source", type=str, required=True, help="Path to UCH-Thermal-Pose root")
    parser.add_argument("--target", type=str, required=True, help="Target root, e.g. data/uch_thermal_pose")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking")
    return parser.parse_args()


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


def rewrite_json(annotation_path: Path, images_root: Path, split_name: str, target_annotation_path: Path, copy_images: bool) -> None:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    updated_images = []
    image_map: dict[int, dict] = {}
    for image_entry in payload.get("images", []):
        src_image = images_root / image_entry["file_name"]
        if not src_image.exists():
            raise FileNotFoundError(src_image)
        target_image = target_annotation_path.parent.parent / "images" / split_name / src_image.name
        ensure_link_or_copy(src_image, target_image, copy_images=copy_images)
        new_entry = dict(image_entry)
        new_entry["file_name"] = f"{split_name}/{src_image.name}"
        updated_images.append(new_entry)
        image_map[int(new_entry["id"])] = new_entry

    payload["images"] = updated_images
    target_annotation_path.parent.mkdir(parents=True, exist_ok=True)
    target_annotation_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{split_name}: {len(updated_images)} images -> {target_annotation_path}")


def main() -> None:
    args = parse_args()
    source_root = Path(args.source)
    target_root = Path(args.target)

    set_a = source_root / "Set-A"
    set_b = source_root / "Set-B"
    rewrite_json(
        annotation_path=set_a / "annotations" / "thermalPose_train.json",
        images_root=set_a / "train",
        split_name="train",
        target_annotation_path=target_root / "annotations" / "train.json",
        copy_images=args.copy_images,
    )
    rewrite_json(
        annotation_path=set_a / "annotations" / "thermalPose_val.json",
        images_root=set_a / "val",
        split_name="val",
        target_annotation_path=target_root / "annotations" / "val.json",
        copy_images=args.copy_images,
    )
    rewrite_json(
        annotation_path=set_b / "annotations" / "thermalPose_test.json",
        images_root=set_b / "test",
        split_name="test",
        target_annotation_path=target_root / "annotations" / "test.json",
        copy_images=args.copy_images,
    )
    print(f"target root: {target_root}")


if __name__ == "__main__":
    main()
