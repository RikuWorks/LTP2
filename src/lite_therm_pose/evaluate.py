from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import ExperimentConfig
from .data import DetectorDataset, PoseDataset
from .data.common import load_coco_records
from .engine import make_loader
from .models.detector import TinyPersonDetector, decode_detections
from .models.pose_topdown import TopDownPoseCNN, decode_pose
from .utils import box_iou_xyxy, ensure_dir, load_image, resize_and_normalize


def evaluate_detector_model(model: TinyPersonDetector, cfg: ExperimentConfig) -> dict[str, float]:
    device = next(model.parameters()).device
    dataset_cfg = cfg.dataset
    if dataset_cfg.val_annotation_file and Path(dataset_cfg.val_annotation_file).exists():
        dataset_cfg = type(cfg.dataset)(**{**cfg.dataset.__dict__, "annotation_file": cfg.dataset.val_annotation_file, "image_root": cfg.dataset.val_image_root or cfg.dataset.image_root})
    dataset = DetectorDataset(dataset_cfg, cfg.detector, cfg.augmentation)
    loader = make_loader(dataset, batch_size=cfg.optim.batch_size, workers=cfg.optim.workers, device=device, shuffle=False)
    model.eval()
    recalls = []
    ious = []
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            gt_boxes = batch["gt_bbox"].to(device)
            preds = decode_detections(model(images), cfg.detector.stride, cfg.detector.score_threshold, cfg.detector.nms_iou_threshold, cfg.detector.max_detections)
            for pred, gt_box in zip(preds, gt_boxes):
                if pred.boxes.numel() == 0:
                    recalls.append(0.0)
                    ious.append(0.0)
                    continue
                iou = box_iou_xyxy(pred.boxes, gt_box.unsqueeze(0)).max().item()
                recalls.append(1.0 if iou >= 0.5 else 0.0)
                ious.append(iou)
    elapsed = time.perf_counter() - start
    count = max(len(dataset), 1)
    return {
        "det_recall50": float(np.mean(recalls) if recalls else 0.0),
        "det_mean_iou": float(np.mean(ious) if ious else 0.0),
        "det_fps": count / max(elapsed, 1e-6),
    }


def _pose_metrics_from_batch(preds: list, batch: dict[str, torch.Tensor]) -> tuple[list[float], list[float]]:
    pck = []
    vis_acc = []
    gt_xy = batch["keypoint_xy"].cpu().numpy()
    gt_visible = batch["keypoint_visible"].cpu().numpy()
    bbox_size = batch["bbox_size"].cpu().numpy()
    for sample_idx, pred in enumerate(preds):
        diag = float(max(bbox_size[sample_idx][0], bbox_size[sample_idx][1], 1.0))
        pred_xy = pred.keypoints.cpu().numpy()
        pred_vis = pred.keypoint_visibility.cpu().numpy()
        dist = np.linalg.norm(pred_xy - gt_xy[sample_idx], axis=1)
        visible_mask = gt_visible[sample_idx] > 0
        if visible_mask.any():
            pck.append(float(np.mean((dist[visible_mask] / diag) <= 0.2)))
        vis_acc.append(float(np.mean((pred_vis >= 0.5) == visible_mask)))
    return pck, vis_acc


def evaluate_pose_model(model: TopDownPoseCNN, cfg: ExperimentConfig) -> dict[str, float]:
    device = next(model.parameters()).device
    dataset_cfg = cfg.dataset
    if dataset_cfg.val_annotation_file and Path(dataset_cfg.val_annotation_file).exists():
        dataset_cfg = type(cfg.dataset)(**{**cfg.dataset.__dict__, "annotation_file": cfg.dataset.val_annotation_file, "image_root": cfg.dataset.val_image_root or cfg.dataset.image_root})
    dataset = PoseDataset(dataset_cfg, cfg.augmentation, train=False)
    loader = make_loader(dataset, batch_size=cfg.optim.batch_size, workers=cfg.optim.workers, device=device, shuffle=False)
    model.eval()
    all_pck = []
    all_vis = []
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            boxes = batch["crop_box"].to(device)
            preds = decode_pose(model(images), boxes, cfg.dataset.image_size, cfg.dataset.body_parts)
            pck, vis_acc = _pose_metrics_from_batch(preds, batch)
            all_pck.extend(pck)
            all_vis.extend(vis_acc)
    elapsed = time.perf_counter() - start
    count = max(len(dataset), 1)
    return {
        "pose_pck20": float(np.mean(all_pck) if all_pck else 0.0),
        "pose_visibility_acc": float(np.mean(all_vis) if all_vis else 0.0),
        "pose_fps": count / max(elapsed, 1e-6),
    }


def evaluate_joint_pipeline(detector: TinyPersonDetector, pose_model: TopDownPoseCNN, cfg: ExperimentConfig) -> dict[str, float]:
    detector_device = next(detector.parameters()).device
    pose_device = next(pose_model.parameters()).device
    dataset_cfg = cfg.dataset
    if dataset_cfg.val_annotation_file and Path(dataset_cfg.val_annotation_file).exists():
        dataset_cfg = type(cfg.dataset)(**{**cfg.dataset.__dict__, "annotation_file": cfg.dataset.val_annotation_file, "image_root": cfg.dataset.val_image_root or cfg.dataset.image_root})
    records = load_coco_records(dataset_cfg)
    detector.eval()
    pose_model.eval()
    pck = []
    start = time.perf_counter()
    with torch.no_grad():
        for record in records:
            image = load_image(record.image_path, grayscale=dataset_cfg.grayscale)
            det_input, scale_x, scale_y = resize_and_normalize(image, cfg.detector.image_size, dataset_cfg.grayscale)
            det_tensor = torch.from_numpy(det_input.transpose(2, 0, 1)).unsqueeze(0).to(detector_device)
            pred = decode_detections(detector(det_tensor), cfg.detector.stride, cfg.detector.score_threshold, cfg.detector.nms_iou_threshold, cfg.detector.max_detections)[0]
            if pred.boxes.numel() == 0:
                pck.append(0.0)
                continue
            gt_box = torch.tensor([[record.bbox[0] * scale_x, record.bbox[1] * scale_y, (record.bbox[0] + record.bbox[2]) * scale_x, (record.bbox[1] + record.bbox[3]) * scale_y]], device=detector_device)
            best_idx = box_iou_xyxy(pred.boxes, gt_box).squeeze(1).argmax()
            box = pred.boxes[best_idx].detach().cpu().numpy()
            x1, y1, x2, y2 = [int(v) for v in [box[0] / scale_x, box[1] / scale_y, box[2] / scale_x, box[3] / scale_y]]
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(max(x2, x1 + 1), image.shape[1])
            y2 = min(max(y2, y1 + 1), image.shape[0])
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                pck.append(0.0)
                continue
            pose_input, _, _ = resize_and_normalize(crop, dataset_cfg.image_size, dataset_cfg.grayscale)
            pose_tensor = torch.from_numpy(pose_input.transpose(2, 0, 1)).unsqueeze(0).to(pose_device)
            pose_preds = decode_pose(
                pose_model(pose_tensor),
                torch.tensor([[x1, y1, x2, y2]], device=pose_device, dtype=torch.float32),
                dataset_cfg.image_size,
                dataset_cfg.body_parts,
            )[0]
            gt_kp = np.array(record.keypoints, dtype=np.float32).reshape(-1, 3)
            pred_xy = pose_preds.keypoints.cpu().numpy()
            gt_xy = gt_kp[:, :2]
            visible = gt_kp[:, 2] > 0
            diag = max(x2 - x1, y2 - y1, 1)
            if visible.any():
                dist = np.linalg.norm(pred_xy - gt_xy, axis=1)
                pck.append(float(np.mean((dist[visible] / diag) <= 0.2)))
    elapsed = time.perf_counter() - start
    count = max(len(records), 1)
    return {"joint_score": float(np.mean(pck) if pck else 0.0), "joint_fps": count / max(elapsed, 1e-6)}


def write_summary_report(rows: list[dict[str, float | str]], output_dir: str | Path) -> None:
    target = ensure_dir(output_dir)
    csv_path = target / "summary.csv"
    if not rows:
        return
    columns = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    md_path = target / "summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(str(row[col]) for col in columns) + " |\n")

    labels = [str(row["model"]) for row in rows]
    det_scores = [float(row["det_recall50"]) for row in rows]
    pose_scores = [float(row["pose_pck20"]) for row in rows]
    joint_scores = [float(row["joint_score"]) for row in rows]
    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(14, 6))
    plt.bar(x - width, det_scores, width=width, label="Detector Recall@0.5")
    plt.bar(x, pose_scores, width=width, label="Pose PCK@0.2")
    plt.bar(x + width, joint_scores, width=width, label="Joint Score")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.legend()
    plt.savefig(target / "summary.png", dpi=160)
    plt.close()
