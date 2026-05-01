from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from lite_therm_pose import load_config
from lite_therm_pose.checkpoints import load_flexible_state_dict
from lite_therm_pose.data.otp2_test import OTP2TestImage, OTP2TestPerson, load_otp2_test_images
from lite_therm_pose.inference import RuntimeBundle, run_topdown_inference
from lite_therm_pose.models.detector import DetectorPrediction, TinyPersonDetector, decode_detections
from lite_therm_pose.models.pose_topdown import PosePrediction, TopDownPoseCNN, decode_pose
from lite_therm_pose.profile import checkpoint_size_mb, parameter_stats, peak_memory_mb, reset_peak_memory
from lite_therm_pose.runtime import resolve_model_device
from lite_therm_pose.utils import box_iou_xyxy, draw_pose, ensure_dir, load_image, resize_and_normalize


@dataclass
class BenchmarkRun:
    run_id: str
    model_name: str
    suite_name: str
    detector_ckpt: Path
    pose_ckpt: Path


HEAD_INDICES = (0, 1, 2, 3, 4)
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark models on OpenThermalPose2 testimage/testlabel.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--suite-dir", type=str, action="append", required=True, help="Suite output directory. Can be passed multiple times.")
    parser.add_argument("--output", type=str, default="outputs/otp2_test_benchmark")
    parser.add_argument("--models", type=str, default="", help="Comma-separated model names to include.")
    parser.add_argument("--test-images-dir", type=str, default="")
    parser.add_argument("--test-labels-dir", type=str, default="")
    parser.add_argument("--search-root", type=str, action="append", default=[], help="Additional roots to search for OTP2 test data.")
    parser.add_argument("--gpu-device", type=str, default="cuda:0")
    parser.add_argument("--cpu-device", type=str, default="cpu")
    parser.add_argument("--visual-samples", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    return parser.parse_args()


def discover_runs(suite_dirs: list[str], models_filter: set[str]) -> list[BenchmarkRun]:
    runs: list[BenchmarkRun] = []
    seen: set[str] = set()
    for suite_dir_str in suite_dirs:
        suite_dir = Path(suite_dir_str)
        if not suite_dir.exists():
            continue
        for model_dir in sorted(path for path in suite_dir.iterdir() if path.is_dir()):
            model_name = model_dir.name
            if models_filter and model_name not in models_filter:
                continue
            detector_ckpt = model_dir / "finetune_detector" / "detector_last.pt"
            pose_ckpt = model_dir / "finetune_pose" / "pose_last.pt"
            if not detector_ckpt.exists() or not pose_ckpt.exists():
                continue
            run_id = model_name
            if run_id in seen:
                run_id = f"{suite_dir.name}__{model_name}"
            seen.add(run_id)
            runs.append(
                BenchmarkRun(
                    run_id=run_id,
                    model_name=model_name,
                    suite_name=suite_dir.name,
                    detector_ckpt=detector_ckpt,
                    pose_ckpt=pose_ckpt,
                )
            )
    return runs


def build_bundle(config_path: str, run: BenchmarkRun, detector_device_name: str, pose_device_name: str, draw_parts: bool = False) -> RuntimeBundle:
    cfg = load_config(config_path)
    detector_model_name = cfg.model.detector_name or run.model_name
    pose_model_name = cfg.model.pose_name or run.model_name
    cfg.model.name = pose_model_name
    cfg.runtime.detector_device = detector_device_name
    cfg.runtime.pose_device = pose_device_name
    cfg.runtime.device = detector_device_name if detector_device_name == pose_device_name else "auto"
    cfg.runtime.draw_parts = draw_parts
    detector_device = resolve_model_device(cfg.runtime.detector_device, cfg.runtime.device, "detector")
    pose_device = resolve_model_device(cfg.runtime.pose_device, cfg.runtime.device, "pose")
    in_channels = 1 if cfg.dataset.grayscale else 3
    detector = TinyPersonDetector(in_channels=in_channels, model_name=detector_model_name).to(detector_device).eval()
    pose_model = TopDownPoseCNN(
        num_keypoints=cfg.dataset.num_keypoints,
        num_parts=len(cfg.dataset.body_parts),
        in_channels=in_channels,
        model_name=pose_model_name,
    ).to(pose_device).eval()
    load_flexible_state_dict(detector, str(run.detector_ckpt))
    load_flexible_state_dict(pose_model, str(run.pose_ckpt))
    return RuntimeBundle(
        detector=detector,
        pose_model=pose_model,
        detector_device=detector_device,
        pose_device=pose_device,
        dataset_cfg=cfg.dataset,
        detector_cfg=cfg.detector,
        draw_parts=draw_parts,
    )


def xyxy_from_xywh(bbox_xywh: list[float]) -> list[float]:
    x, y, w, h = bbox_xywh
    return [x, y, x + w, y + h]


def xywh_from_xyxy(bbox_xyxy: list[float]) -> list[float]:
    x1, y1, x2, y2 = bbox_xyxy
    return [float(x1), float(y1), float(max(x2 - x1, 1.0)), float(max(y2 - y1, 1.0))]


def head_size(person: OTP2TestPerson) -> float:
    visible_head = person.keypoints[list(HEAD_INDICES)]
    visible_head = visible_head[visible_head[:, 2] > 0]
    if visible_head.shape[0] >= 2:
        width = float(visible_head[:, 0].max() - visible_head[:, 0].min())
        height = float(visible_head[:, 1].max() - visible_head[:, 1].min())
        return max(width, height, 1.0)
    bbox_w = float(person.bbox_xywh[2])
    bbox_h = float(person.bbox_xywh[3])
    return max(0.3 * max(bbox_w, bbox_h), 1.0)


def person_metrics(pred: PosePrediction | None, person: OTP2TestPerson) -> dict[str, float]:
    gt = person.keypoints
    visible = gt[:, 2] > 0
    bbox_norm = max(float(person.bbox_xywh[2]), float(person.bbox_xywh[3]), 1.0)
    head_norm = head_size(person)
    if pred is None or not visible.any():
        return {
            "pck20": 0.0,
            "pck10": 0.0,
            "pck05": 0.0,
            "pckh50": 0.0,
            "pckh30": 0.0,
            "mean_error_px": 0.0,
            "mean_error_bbox": 0.0,
            "visibility_acc": 0.0,
        }
    pred_xy = pred.keypoints.cpu().numpy()
    pred_vis = pred.keypoint_visibility.cpu().numpy()
    gt_xy = gt[:, :2]
    dist = np.linalg.norm(pred_xy - gt_xy, axis=1)
    dist_visible = dist[visible]
    pred_visible = pred_vis >= 0.5
    return {
        "pck20": float(np.mean((dist_visible / bbox_norm) <= 0.2)),
        "pck10": float(np.mean((dist_visible / bbox_norm) <= 0.1)),
        "pck05": float(np.mean((dist_visible / bbox_norm) <= 0.05)),
        "pckh50": float(np.mean((dist_visible / head_norm) <= 0.5)),
        "pckh30": float(np.mean((dist_visible / head_norm) <= 0.3)),
        "mean_error_px": float(np.mean(dist_visible)),
        "mean_error_bbox": float(np.mean(dist_visible / bbox_norm)),
        "visibility_acc": float(np.mean(pred_visible == visible)),
    }


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_coco_gt(test_images: list[OTP2TestImage]) -> COCO:
    dataset = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "keypoints": COCO_KEYPOINT_NAMES,
                "skeleton": COCO_SKELETON,
            }
        ],
    }
    ann_id = 1
    for item in test_images:
        dataset["images"].append(
            {
                "id": item.image_id,
                "file_name": item.image_path.name,
                "width": item.width,
                "height": item.height,
            }
        )
        for person in item.persons:
            keypoints = person.keypoints.reshape(-1, 3)
            flat_keypoints = []
            num_keypoints = 0
            for px, py, vis in keypoints.tolist():
                if vis > 0:
                    flat_keypoints.extend([float(px), float(py), float(vis)])
                    num_keypoints += 1
                else:
                    flat_keypoints.extend([0.0, 0.0, 0.0])
            bbox = [float(value) for value in person.bbox_xywh]
            dataset["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": item.image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "area": float(bbox[2] * bbox[3]),
                    "iscrowd": 0,
                    "num_keypoints": num_keypoints,
                    "keypoints": flat_keypoints,
                }
            )
            ann_id += 1
    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()
    return coco


def make_empty_coco_results(image_id: int) -> list[dict[str, float | int | list[float]]]:
    return [{"image_id": image_id, "category_id": 1, "bbox": [0.0, 0.0, 1.0, 1.0], "score": 1.0e-9}]


def make_empty_keypoint_results(image_id: int) -> list[dict[str, float | int | list[float]]]:
    return [
        {
            "image_id": image_id,
            "category_id": 1,
            "keypoints": [0.0] * (17 * 3),
            "score": 1.0e-9,
        }
    ]


def _run_cocoeval(coco_gt: COCO, results: list[dict], iou_type: str) -> tuple[COCOeval, dict[str, float]]:
    coco_dt = coco_gt.loadRes(results)
    evaluator = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    if iou_type == "bbox":
        prefix = "det"
    else:
        prefix = "pose"
    return evaluator, {
        f"{prefix}_map50_95_test": float(evaluator.stats[0]),
        f"{prefix}_map50_test": float(evaluator.stats[1]),
        f"{prefix}_map75_test": float(evaluator.stats[2]),
        f"{prefix}_ar50_95_test": float(evaluator.stats[8]),
    }


def compute_mean_oks(test_images: list[OTP2TestImage], joint_pose_predictions: dict[int, list[PosePrediction]]) -> float:
    oks_values: list[float] = []
    sigmas = np.array([0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089], dtype=np.float32)
    vars_ = (sigmas * 2.0) ** 2
    for item in test_images:
        preds = joint_pose_predictions.get(item.image_id, [])
        for gt_idx, person in enumerate(item.persons):
            if gt_idx >= len(preds):
                oks_values.append(0.0)
                continue
            pred = preds[gt_idx]
            gt = person.keypoints
            visible = gt[:, 2] > 0
            if not visible.any():
                continue
            pred_xy = pred.keypoints.cpu().numpy()
            gt_xy = gt[:, :2]
            bbox_area = max(float(person.bbox_xywh[2] * person.bbox_xywh[3]), 1.0)
            squared_dist = np.sum((pred_xy - gt_xy) ** 2, axis=1)
            oks = np.exp(-squared_dist / (2.0 * bbox_area * vars_))
            oks_values.append(float(np.mean(oks[visible])))
    return float(np.mean(oks_values) if oks_values else 0.0)


@torch.no_grad()
def predict_detector(bundle: RuntimeBundle, image: np.ndarray) -> DetectorPrediction:
    detector_input, scale_x, scale_y = resize_and_normalize(image, bundle.detector_cfg.image_size, bundle.dataset_cfg.grayscale)
    det_tensor = torch.from_numpy(detector_input.transpose(2, 0, 1)).unsqueeze(0).to(bundle.detector_device)
    pred = decode_detections(
        bundle.detector(det_tensor),
        stride=bundle.detector_cfg.stride,
        score_threshold=bundle.detector_cfg.score_threshold,
        nms_iou_threshold=bundle.detector_cfg.nms_iou_threshold,
        max_detections=bundle.detector_cfg.max_detections,
    )[0]
    if pred.boxes.numel() == 0:
        return DetectorPrediction(boxes=pred.boxes.cpu(), scores=pred.scores.cpu())
    boxes = pred.boxes.detach().cpu().clone()
    boxes[:, [0, 2]] /= max(scale_x, 1.0e-6)
    boxes[:, [1, 3]] /= max(scale_y, 1.0e-6)
    return DetectorPrediction(boxes=boxes, scores=pred.scores.detach().cpu())


@torch.no_grad()
def predict_pose(bundle: RuntimeBundle, image: np.ndarray, crop_boxes_xyxy: list[list[float]]) -> list[PosePrediction]:
    crops = []
    crop_boxes = []
    for x1, y1, x2, y2 in crop_boxes_xyxy:
        ix1 = max(int(round(x1)), 0)
        iy1 = max(int(round(y1)), 0)
        ix2 = min(max(int(round(x2)), ix1 + 1), image.shape[1])
        iy2 = min(max(int(round(y2)), iy1 + 1), image.shape[0])
        crop = image[iy1:iy2, ix1:ix2]
        if crop.size == 0:
            continue
        crop_norm, _, _ = resize_and_normalize(crop, bundle.dataset_cfg.image_size, bundle.dataset_cfg.grayscale)
        crops.append(crop_norm.transpose(2, 0, 1))
        crop_boxes.append([ix1, iy1, ix2, iy2])
    if not crops:
        return []
    crop_tensor = torch.from_numpy(np.stack(crops)).to(bundle.pose_device)
    preds = bundle.pose_model(crop_tensor)
    return decode_pose(
        preds,
        crop_boxes=torch.tensor(crop_boxes, device=bundle.pose_device, dtype=torch.float32),
        image_size=bundle.dataset_cfg.image_size,
        body_parts=bundle.dataset_cfg.body_parts,
    )


def evaluate_detector(bundle: RuntimeBundle, test_images: list[OTP2TestImage]) -> dict[str, float]:
    recall50 = []
    precision50 = []
    mean_iou = []
    for item in test_images:
        image = load_image(item.image_path, grayscale=bundle.dataset_cfg.grayscale)
        pred = predict_detector(bundle, image)
        gt_boxes = torch.tensor([xyxy_from_xywh(person.bbox_xywh) for person in item.persons], dtype=torch.float32)
        if pred.boxes.numel() == 0:
            recall50.extend([0.0] * len(item.persons))
            mean_iou.extend([0.0] * len(item.persons))
            precision50.append(0.0)
            continue
        iou = box_iou_xyxy(pred.boxes.float(), gt_boxes.float())
        best_iou_per_gt = iou.max(dim=0).values
        best_iou_per_pred = iou.max(dim=1).values
        recall50.extend((best_iou_per_gt >= 0.5).float().tolist())
        mean_iou.extend(best_iou_per_gt.tolist())
        precision50.append(float((best_iou_per_pred >= 0.5).float().mean().item()))
    return {
        "det_recall50_test": float(np.mean(recall50) if recall50 else 0.0),
        "det_precision50_test": float(np.mean(precision50) if precision50 else 0.0),
        "det_mean_iou_test": float(np.mean(mean_iou) if mean_iou else 0.0),
    }


def evaluate_pose_gt(bundle: RuntimeBundle, test_images: list[OTP2TestImage]) -> dict[str, float]:
    metrics: dict[str, list[float]] = {name: [] for name in ["pck20", "pck10", "pck05", "pckh50", "pckh30", "mean_error_px", "mean_error_bbox", "visibility_acc"]}
    for item in test_images:
        image = load_image(item.image_path, grayscale=bundle.dataset_cfg.grayscale)
        boxes = [xyxy_from_xywh(person.bbox_xywh) for person in item.persons]
        preds = predict_pose(bundle, image, boxes)
        for index, person in enumerate(item.persons):
            pred = preds[index] if index < len(preds) else None
            person_result = person_metrics(pred, person)
            for key, value in person_result.items():
                metrics[key].append(value)
    return {
        "pose_pck20_test": float(np.mean(metrics["pck20"]) if metrics["pck20"] else 0.0),
        "pose_pck10_test": float(np.mean(metrics["pck10"]) if metrics["pck10"] else 0.0),
        "pose_pck05_test": float(np.mean(metrics["pck05"]) if metrics["pck05"] else 0.0),
        "pose_pckh50_test": float(np.mean(metrics["pckh50"]) if metrics["pckh50"] else 0.0),
        "pose_pckh30_test": float(np.mean(metrics["pckh30"]) if metrics["pckh30"] else 0.0),
        "pose_mean_error_px_test": float(np.mean(metrics["mean_error_px"]) if metrics["mean_error_px"] else 0.0),
        "pose_mean_error_bbox_test": float(np.mean(metrics["mean_error_bbox"]) if metrics["mean_error_bbox"] else 0.0),
        "pose_visibility_acc_test": float(np.mean(metrics["visibility_acc"]) if metrics["visibility_acc"] else 0.0),
    }


def evaluate_joint(bundle: RuntimeBundle, test_images: list[OTP2TestImage]) -> dict[str, float]:
    metrics: dict[str, list[float]] = {name: [] for name in ["pck20", "pck10", "pck05", "pckh50", "pckh30", "mean_error_px", "mean_error_bbox", "visibility_acc"]}
    joint_det_iou = []
    for item in test_images:
        image = load_image(item.image_path, grayscale=bundle.dataset_cfg.grayscale)
        pred = predict_detector(bundle, image)
        gt_boxes = torch.tensor([xyxy_from_xywh(person.bbox_xywh) for person in item.persons], dtype=torch.float32)
        if pred.boxes.numel() == 0:
            for person in item.persons:
                result = person_metrics(None, person)
                for key, value in result.items():
                    metrics[key].append(value)
                joint_det_iou.append(0.0)
            continue
        iou = box_iou_xyxy(pred.boxes.float(), gt_boxes.float())
        best_iou_per_gt, best_idx_per_gt = iou.max(dim=0)
        matched_boxes = [pred.boxes[int(index)].tolist() for index in best_idx_per_gt]
        pose_preds = predict_pose(bundle, image, matched_boxes)
        for index, person in enumerate(item.persons):
            pred_pose = pose_preds[index] if index < len(pose_preds) else None
            det_iou = float(best_iou_per_gt[index].item())
            result = person_metrics(pred_pose, person)
            for key, value in result.items():
                metrics[key].append(value)
            joint_det_iou.append(det_iou)
    return {
        "joint_score_test": float(np.mean(metrics["pck20"]) if metrics["pck20"] else 0.0),
        "joint_pckh50_test": float(np.mean(metrics["pckh50"]) if metrics["pckh50"] else 0.0),
        "joint_mean_iou_test": float(np.mean(joint_det_iou) if joint_det_iou else 0.0),
    }


def collect_coco_predictions(bundle: RuntimeBundle, test_images: list[OTP2TestImage]) -> tuple[list[dict], list[dict], dict[int, list[PosePrediction]]]:
    bbox_results: list[dict] = []
    keypoint_results: list[dict] = []
    joint_pose_predictions: dict[int, list[PosePrediction]] = {}
    for item in test_images:
        image = load_image(item.image_path, grayscale=bundle.dataset_cfg.grayscale)
        det_pred = predict_detector(bundle, image)
        gt_boxes = torch.tensor([xyxy_from_xywh(person.bbox_xywh) for person in item.persons], dtype=torch.float32)
        if det_pred.boxes.numel() > 0:
            iou = box_iou_xyxy(det_pred.boxes.float(), gt_boxes.float())
            best_idx_per_gt = iou.max(dim=0).indices
            matched_boxes = [det_pred.boxes[int(index)].tolist() for index in best_idx_per_gt]
            pose_preds = predict_pose(bundle, image, matched_boxes)
            joint_pose_predictions[item.image_id] = pose_preds
            for box, score in zip(det_pred.boxes.tolist(), det_pred.scores.tolist()):
                bbox_results.append(
                    {
                        "image_id": item.image_id,
                        "category_id": 1,
                        "bbox": xywh_from_xyxy(box),
                        "score": float(score),
                    }
                )
            for pred_idx, pose_pred in enumerate(pose_preds):
                if pred_idx >= len(matched_boxes):
                    continue
                det_score = float(det_pred.scores[int(best_idx_per_gt[pred_idx])].item())
                flattened_keypoints: list[float] = []
                keypoint_scores = pose_pred.keypoint_scores.tolist()
                for (px, py), kp_score in zip(pose_pred.keypoints.tolist(), keypoint_scores):
                    vis_value = 2.0 if kp_score > 0.15 else 0.0
                    flattened_keypoints.extend([float(px), float(py), vis_value])
                keypoint_results.append(
                    {
                        "image_id": item.image_id,
                        "category_id": 1,
                        "keypoints": flattened_keypoints,
                        "score": float(det_score * np.mean(keypoint_scores)),
                    }
                )
        else:
            joint_pose_predictions[item.image_id] = []
    if not bbox_results:
        bbox_results = make_empty_coco_results(test_images[0].image_id)
    if not keypoint_results:
        keypoint_results = make_empty_keypoint_results(test_images[0].image_id)
    return bbox_results, keypoint_results, joint_pose_predictions


def benchmark_runtime(bundle: RuntimeBundle, test_images: list[OTP2TestImage], warmup: int) -> dict[str, float]:
    unique_images = test_images
    detector_device = bundle.detector_device
    pose_device = bundle.pose_device
    warmup_images = unique_images[: min(warmup, len(unique_images))]

    for item in warmup_images:
        image = load_image(item.image_path, grayscale=bundle.dataset_cfg.grayscale)
        _ = predict_detector(bundle, image)
        _ = predict_pose(bundle, image, [xyxy_from_xywh(person.bbox_xywh) for person in item.persons])
        _ = run_topdown_inference(bundle, image)

    det_latencies = []
    reset_peak_memory(detector_device)
    sync_device(detector_device)
    det_start = time.perf_counter()
    for item in unique_images:
        image = load_image(item.image_path, grayscale=bundle.dataset_cfg.grayscale)
        sync_device(detector_device)
        t0 = time.perf_counter()
        _ = predict_detector(bundle, image)
        sync_device(detector_device)
        det_latencies.append(time.perf_counter() - t0)
    sync_device(detector_device)
    det_elapsed = time.perf_counter() - det_start

    pose_latencies = []
    total_persons = sum(len(item.persons) for item in unique_images)
    reset_peak_memory(pose_device)
    sync_device(pose_device)
    pose_start = time.perf_counter()
    for item in unique_images:
        image = load_image(item.image_path, grayscale=bundle.dataset_cfg.grayscale)
        boxes = [xyxy_from_xywh(person.bbox_xywh) for person in item.persons]
        sync_device(pose_device)
        t0 = time.perf_counter()
        _ = predict_pose(bundle, image, boxes)
        sync_device(pose_device)
        pose_latencies.append(time.perf_counter() - t0)
    sync_device(pose_device)
    pose_elapsed = time.perf_counter() - pose_start

    joint_latencies = []
    reset_peak_memory(detector_device)
    if pose_device != detector_device:
        reset_peak_memory(pose_device)
    sync_device(detector_device)
    if pose_device != detector_device:
        sync_device(pose_device)
    joint_start = time.perf_counter()
    for item in unique_images:
        image = load_image(item.image_path, grayscale=bundle.dataset_cfg.grayscale)
        sync_device(detector_device)
        if pose_device != detector_device:
            sync_device(pose_device)
        t0 = time.perf_counter()
        _ = run_topdown_inference(bundle, image)
        sync_device(detector_device)
        if pose_device != detector_device:
            sync_device(pose_device)
        joint_latencies.append(time.perf_counter() - t0)
    sync_device(detector_device)
    if pose_device != detector_device:
        sync_device(pose_device)
    joint_elapsed = time.perf_counter() - joint_start

    image_count = max(len(unique_images), 1)
    person_count = max(total_persons, 1)
    return {
        "det_fps_images": image_count / max(det_elapsed, 1.0e-6),
        "det_latency_ms_test": 1000.0 * float(np.mean(det_latencies) if det_latencies else 0.0),
        "det_peak_memory_mb_test": peak_memory_mb(detector_device),
        "pose_fps_images": image_count / max(pose_elapsed, 1.0e-6),
        "pose_fps_persons": person_count / max(pose_elapsed, 1.0e-6),
        "pose_latency_ms_test": 1000.0 * float(np.mean(pose_latencies) if pose_latencies else 0.0),
        "pose_peak_memory_mb_test": peak_memory_mb(pose_device),
        "joint_fps_images": image_count / max(joint_elapsed, 1.0e-6),
        "joint_latency_ms_test": 1000.0 * float(np.mean(joint_latencies) if joint_latencies else 0.0),
        "joint_peak_memory_mb_test": max(peak_memory_mb(detector_device), peak_memory_mb(pose_device)),
    }


def render_ground_truth(bundle: RuntimeBundle, item: OTP2TestImage) -> np.ndarray:
    image = load_image(item.image_path, grayscale=bundle.dataset_cfg.grayscale)
    visual = image.copy()
    if visual.ndim == 3 and visual.shape[2] == 1:
        visual = cv2.cvtColor(visual, cv2.COLOR_GRAY2BGR)
    for person in item.persons:
        x1, y1, x2, y2 = xyxy_from_xywh(person.bbox_xywh)
        cv2.rectangle(visual, (int(x1), int(y1)), (int(x2), int(y2)), (0, 100, 255), 2)
        keypoints = [(float(x), float(y), float(v)) for x, y, v in person.keypoints.tolist()]
        visual = draw_pose(visual, keypoints, None)
    return visual


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    canvas = cv2.copyMakeBorder(image, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    return canvas


def resize_panel(image: np.ndarray, width: int = 360) -> np.ndarray:
    height = int(round(image.shape[0] * (width / image.shape[1])))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def compose_grid(panels: list[np.ndarray], columns: int = 3) -> np.ndarray:
    rows = []
    blank = np.full_like(panels[0], 255)
    for start in range(0, len(panels), columns):
        row_panels = panels[start : start + columns]
        while len(row_panels) < columns:
            row_panels.append(blank.copy())
        rows.append(np.hstack(row_panels))
    return np.vstack(rows)


def save_qualitative_examples(
    output_dir: Path,
    config_path: str,
    visual_bundle: RuntimeBundle,
    runs: list[BenchmarkRun],
    test_images: list[OTP2TestImage],
    num_samples: int,
) -> None:
    if not test_images:
        return
    qualitative_dir = ensure_dir(output_dir / "qualitative")
    selected = test_images[: min(num_samples, len(test_images))]
    for item in selected:
        image = load_image(item.image_path, grayscale=visual_bundle.dataset_cfg.grayscale)
        gt_visual = add_title(resize_panel(render_ground_truth(visual_bundle, item)), "ground truth")
        panels = [gt_visual]
        for run in runs:
            bundle = build_bundle(
                config_path=config_path,
                run=run,
                detector_device_name=str(visual_bundle.detector_device),
                pose_device_name=str(visual_bundle.pose_device),
                draw_parts=False,
            )
            visual = run_topdown_inference(bundle, image)
            panels.append(add_title(resize_panel(visual), run.run_id))
            model_dir = ensure_dir(qualitative_dir / run.run_id)
            cv2.imwrite(str(model_dir / item.image_path.name), visual)
        sheet = compose_grid(panels, columns=3)
        cv2.imwrite(str(qualitative_dir / f"{item.image_path.stem}_comparison.png"), sheet)


def write_rows(rows: list[dict[str, float | str]], output_dir: Path) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(str(row[col]) for col in columns) + " |\n")


def plot_charts(rows: list[dict[str, float | str]], output_dir: Path) -> None:
    labels = [str(row["run_id"]) for row in rows]
    x = np.arange(len(labels))

    plt.figure(figsize=(15, 6))
    width = 0.22
    plt.bar(x - width, [float(row["det_recall50_test"]) for row in rows], width=width, label="det_recall50")
    plt.bar(x, [float(row["pose_pck20_test"]) for row in rows], width=width, label="pose_pck20")
    plt.bar(x + width, [float(row["pose_pckh50_test"]) for row in rows], width=width, label="pose_pckh50")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.legend()
    plt.savefig(output_dir / "accuracy_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(15, 6))
    width = 0.22
    plt.bar(x - width, [float(row["det_map50_95_test"]) for row in rows], width=width, label="det mAP50-95")
    plt.bar(x, [float(row["pose_map50_95_test"]) for row in rows], width=width, label="pose mAP50-95")
    plt.bar(x + width, [float(row["pose_mean_oks_test"]) for row in rows], width=width, label="mean OKS")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.legend()
    plt.savefig(output_dir / "coco_metrics_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(15, 6))
    width = 0.35
    plt.bar(x - width / 2, [float(row["cpu_joint_fps_images"]) for row in rows], width=width, label="CPU joint fps")
    plt.bar(x + width / 2, [float(row["gpu_joint_fps_images"]) for row in rows], width=width, label="GPU joint fps")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.tight_layout()
    plt.legend()
    plt.savefig(output_dir / "speed_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 7))
    plt.scatter(
        [float(row["total_params_m"]) for row in rows],
        [float(row["joint_score_test"]) for row in rows],
        s=[float(row["gpu_joint_peak_memory_mb_test"]) / 2.0 + 20.0 for row in rows],
        alpha=0.8,
        c=np.linspace(0.2, 0.9, len(rows)),
        cmap="viridis",
    )
    for row in rows:
        plt.annotate(str(row["run_id"]), (float(row["total_params_m"]), float(row["joint_score_test"])), fontsize=9)
    plt.xlabel("Total params (M)")
    plt.ylabel("Joint score on OTP2 test")
    plt.tight_layout()
    plt.savefig(output_dir / "params_vs_joint_score.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output)
    models_filter = {item.strip() for item in args.models.split(",") if item.strip()}
    runs = discover_runs(args.suite_dir, models_filter)
    if not runs:
        raise FileNotFoundError("No runnable model checkpoints were found in the suite directories.")
    test_images = load_otp2_test_images(
        images_dir=args.test_images_dir or None,
        labels_dir=args.test_labels_dir or None,
        search_roots=args.search_root or None,
    )
    coco_gt = build_coco_gt(test_images)
    rows: list[dict[str, float | str]] = []
    gpu_available = torch.cuda.is_available()
    visual_device = args.gpu_device if gpu_available else args.cpu_device
    visual_bundle = build_bundle(args.config, runs[0], visual_device, visual_device, draw_parts=False)

    for run in runs:
        eval_device = args.gpu_device if gpu_available else args.cpu_device
        eval_bundle = build_bundle(args.config, run, eval_device, eval_device, draw_parts=False)
        detector_stats = evaluate_detector(eval_bundle, test_images)
        pose_stats = evaluate_pose_gt(eval_bundle, test_images)
        joint_stats = evaluate_joint(eval_bundle, test_images)
        bbox_results, keypoint_results, joint_pose_predictions = collect_coco_predictions(eval_bundle, test_images)
        _, det_coco_stats = _run_cocoeval(coco_gt, bbox_results, "bbox")
        _, pose_coco_stats = _run_cocoeval(coco_gt, keypoint_results, "keypoints")
        pose_mean_oks = compute_mean_oks(test_images, joint_pose_predictions)

        cpu_bundle = build_bundle(args.config, run, args.cpu_device, args.cpu_device, draw_parts=False)
        cpu_speed = benchmark_runtime(cpu_bundle, test_images, warmup=args.warmup)

        if gpu_available:
            gpu_bundle = build_bundle(args.config, run, args.gpu_device, args.gpu_device, draw_parts=False)
            gpu_speed = benchmark_runtime(gpu_bundle, test_images, warmup=args.warmup)
        else:
            gpu_speed = {}

        stats = {
            "run_id": run.run_id,
            "suite_name": run.suite_name,
            "model_name": run.model_name,
            **parameter_stats(eval_bundle.detector, "detector"),
            **parameter_stats(eval_bundle.pose_model, "pose"),
            "total_params_m": parameter_stats(eval_bundle.detector, "detector")["detector_params_m"] + parameter_stats(eval_bundle.pose_model, "pose")["pose_params_m"],
            "detector_checkpoint_mb": checkpoint_size_mb(run.detector_ckpt),
            "pose_checkpoint_mb": checkpoint_size_mb(run.pose_ckpt),
            **detector_stats,
            **pose_stats,
            **joint_stats,
            **det_coco_stats,
            **pose_coco_stats,
            "pose_mean_oks_test": pose_mean_oks,
            "cpu_det_fps_images": cpu_speed["det_fps_images"],
            "cpu_det_latency_ms_test": cpu_speed["det_latency_ms_test"],
            "cpu_pose_fps_images": cpu_speed["pose_fps_images"],
            "cpu_pose_fps_persons": cpu_speed["pose_fps_persons"],
            "cpu_pose_latency_ms_test": cpu_speed["pose_latency_ms_test"],
            "cpu_joint_fps_images": cpu_speed["joint_fps_images"],
            "cpu_joint_latency_ms_test": cpu_speed["joint_latency_ms_test"],
            "gpu_det_fps_images": gpu_speed.get("det_fps_images", 0.0),
            "gpu_det_latency_ms_test": gpu_speed.get("det_latency_ms_test", 0.0),
            "gpu_det_peak_memory_mb_test": gpu_speed.get("det_peak_memory_mb_test", 0.0),
            "gpu_pose_fps_images": gpu_speed.get("pose_fps_images", 0.0),
            "gpu_pose_fps_persons": gpu_speed.get("pose_fps_persons", 0.0),
            "gpu_pose_latency_ms_test": gpu_speed.get("pose_latency_ms_test", 0.0),
            "gpu_pose_peak_memory_mb_test": gpu_speed.get("pose_peak_memory_mb_test", 0.0),
            "gpu_joint_fps_images": gpu_speed.get("joint_fps_images", 0.0),
            "gpu_joint_latency_ms_test": gpu_speed.get("joint_latency_ms_test", 0.0),
            "gpu_joint_peak_memory_mb_test": gpu_speed.get("joint_peak_memory_mb_test", 0.0),
            "test_images_count": len(test_images),
            "test_persons_count": sum(len(item.persons) for item in test_images),
        }
        rows.append(stats)
        model_dir = ensure_dir(output_dir / run.run_id)
        with (model_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, ensure_ascii=False, indent=2)

    rows.sort(key=lambda row: float(row["joint_score_test"]), reverse=True)
    write_rows(rows, output_dir)
    plot_charts(rows, output_dir)
    save_qualitative_examples(output_dir, args.config, visual_bundle, runs, test_images, num_samples=args.visual_samples)


if __name__ == "__main__":
    main()
