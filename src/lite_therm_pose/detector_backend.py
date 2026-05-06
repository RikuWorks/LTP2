from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .config import DatasetConfig, ExperimentConfig
from .data.common import load_coco_records
from .models.detector import DetectorPrediction, TinyPersonDetector, decode_detections
from .runtime import resolve_model_device
from .utils import ensure_dir, resize_and_normalize


DEFAULT_YOLOV5_REPO_URL = "https://github.com/ultralytics/yolov5.git"


def uses_yolov5_backend(cfg: ExperimentConfig) -> bool:
    return cfg.detector.backend.strip().lower() == "yolov5n"


def ensure_yolov5_repo(repo_dir: str | Path, repo_url: str = DEFAULT_YOLOV5_REPO_URL) -> Path:
    target = Path(repo_dir)
    train_py = target / "train.py"
    hubconf_py = target / "hubconf.py"
    if train_py.exists() and hubconf_py.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and any(target.iterdir()):
        raise FileNotFoundError(
            f"YOLOv5 repository was not found in '{target}'. Either provide a valid repo path or clear this directory."
        )
    git_bin = shutil.which("git")
    if git_bin is None:
        raise FileNotFoundError(
            "git executable was not found while trying to auto-clone YOLOv5. "
            f"Install git or pass --yolov5-repo with an existing clone. Target: {target}"
        )
    try:
        subprocess.run([git_bin, "clone", repo_url, str(target)], check=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Failed to clone YOLOv5 into '{target}'. git executable path: {git_bin}"
        ) from exc
    if not train_py.exists() or not hubconf_py.exists():
        raise FileNotFoundError(
            f"YOLOv5 clone into '{target}' did not produce train.py/hubconf.py. "
            "Check network access or provide --yolov5-repo manually."
        )
    return target


def resolve_yolov5_repo_dir(cfg: ExperimentConfig, workspace_hint: str | Path | None = None) -> Path:
    raw = cfg.detector.yolov5_repo.strip()
    if raw:
        return ensure_yolov5_repo(raw)
    base = Path(workspace_hint) if workspace_hint is not None else Path.cwd()
    auto_dir = base / "external" / "yolov5"
    return ensure_yolov5_repo(auto_dir)


class YOLOv5DetectorAdapter:
    def __init__(
        self,
        repo_dir: str | Path,
        weights_path: str | Path,
        device: torch.device,
        detector_cfg: Any,
    ) -> None:
        self.repo_dir = ensure_yolov5_repo(repo_dir)
        self.weights_path = Path(weights_path)
        self.device = device
        self.detector_cfg = detector_cfg
        self.model = torch.hub.load(
            str(self.repo_dir),
            "custom",
            path=str(self.weights_path),
            source="local",
        )
        self.model.to(device)
        self.model.eval()
        self.parameter_source = getattr(self.model, "model", self.model)

    def eval(self) -> "YOLOv5DetectorAdapter":
        self.model.eval()
        return self

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> DetectorPrediction:
        if image.ndim == 3 and image.shape[2] == 1:
            image_rgb = np.repeat(image, 3, axis=2)
        elif image.ndim == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model(image_rgb, size=int(self.detector_cfg.yolov5_imgsz))
        raw = results.xyxy[0].detach().cpu()
        if raw.numel() == 0:
            return DetectorPrediction(boxes=torch.zeros((0, 4)), scores=torch.zeros(0))
        scores = raw[:, 4]
        keep = scores >= float(self.detector_cfg.score_threshold)
        raw = raw[keep]
        if raw.numel() == 0:
            return DetectorPrediction(boxes=torch.zeros((0, 4)), scores=torch.zeros(0))
        return DetectorPrediction(boxes=raw[:, :4], scores=raw[:, 4])


def detector_parameter_source(detector: Any) -> torch.nn.Module:
    source = getattr(detector, "parameter_source", detector)
    return source


def detector_device(detector: Any) -> torch.device:
    if hasattr(detector, "device"):
        return detector.device
    return next(detector.parameters()).device


def build_detector_runtime(
    cfg: ExperimentConfig,
    checkpoint_path: str | Path,
    explicit_device: str = "",
) -> tuple[Any, torch.device]:
    device = resolve_model_device(explicit_device or cfg.runtime.detector_device, cfg.runtime.device, "detector")
    if uses_yolov5_backend(cfg):
        repo_dir = resolve_yolov5_repo_dir(cfg, workspace_hint=Path(checkpoint_path).parent.parent)
        detector = YOLOv5DetectorAdapter(
            repo_dir=repo_dir,
            weights_path=checkpoint_path,
            device=device,
            detector_cfg=cfg.detector,
        )
        return detector, device
    in_channels = 1 if cfg.dataset.grayscale else 3
    model_name = cfg.model.detector_name or cfg.model.name
    detector = TinyPersonDetector(in_channels=in_channels, model_name=model_name).to(device).eval()
    state = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(state, dict) and "model" in state:
        detector.load_state_dict(state["model"], strict=False)
    else:
        detector.load_state_dict(state, strict=False)
    return detector, device


@torch.no_grad()
def predict_detector_image(detector: Any, cfg: ExperimentConfig, image: np.ndarray) -> DetectorPrediction:
    if hasattr(detector, "predict"):
        return detector.predict(image)
    detector_input, scale_x, scale_y = resize_and_normalize(image, cfg.detector.image_size, cfg.dataset.grayscale)
    det_tensor = torch.from_numpy(detector_input.transpose(2, 0, 1)).unsqueeze(0).to(detector_device(detector))
    pred = decode_detections(
        detector(det_tensor),
        stride=cfg.detector.stride,
        score_threshold=cfg.detector.score_threshold,
        nms_iou_threshold=cfg.detector.nms_iou_threshold,
        max_detections=cfg.detector.max_detections,
    )[0]
    if pred.boxes.numel() == 0:
        return DetectorPrediction(boxes=pred.boxes.detach().cpu(), scores=pred.scores.detach().cpu())
    boxes = pred.boxes.detach().cpu().clone()
    boxes[:, [0, 2]] /= max(scale_x, 1.0e-6)
    boxes[:, [1, 3]] /= max(scale_y, 1.0e-6)
    return DetectorPrediction(boxes=boxes, scores=pred.scores.detach().cpu())


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    source = src.resolve()
    try:
        dst.symlink_to(source)
    except OSError:
        shutil.copy2(source, dst)


def _export_split_to_yolo(dataset_cfg: DatasetConfig, split_name: str, root: Path) -> tuple[Path, Path]:
    records = load_coco_records(dataset_cfg)
    images_dir = ensure_dir(root / "images" / split_name)
    labels_dir = ensure_dir(root / "labels" / split_name)
    grouped: dict[tuple[int, str], list] = defaultdict(list)
    for record in records:
        grouped[(record.image_id, str(record.image_path))].append(record)
    list_path = root / f"{split_name}.txt"
    with list_path.open("w", encoding="utf-8") as listing:
        for (image_id, image_path_str), items in grouped.items():
            src = Path(image_path_str)
            image_name = f"{image_id}_{src.name}"
            dst_image = images_dir / image_name
            _link_or_copy(src, dst_image)
            listing.write(str(dst_image.resolve()) + "\n")
            image = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(src)
            height, width = image.shape[:2]
            label_path = labels_dir / f"{Path(image_name).stem}.txt"
            with label_path.open("w", encoding="utf-8") as handle:
                for item in items:
                    x, y, w, h = item.bbox
                    cx = (x + w / 2.0) / max(width, 1)
                    cy = (y + h / 2.0) / max(height, 1)
                    nw = w / max(width, 1)
                    nh = h / max(height, 1)
                    handle.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
    return images_dir, list_path


def export_detector_dataset_to_yolov5(cfg: ExperimentConfig, output_dir: str | Path) -> Path:
    root = ensure_dir(Path(output_dir) / "yolov5_dataset")
    _export_split_to_yolo(cfg.dataset, "train", root)
    val_cfg = DatasetConfig(**{**cfg.dataset.__dict__})
    if val_cfg.val_annotation_file:
        val_cfg.annotation_file = val_cfg.val_annotation_file
    if val_cfg.val_image_root:
        val_cfg.image_root = val_cfg.val_image_root
    _export_split_to_yolo(val_cfg, "val", root)
    dataset_yaml = root / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {root.resolve()}",
                "train: train.txt",
                "val: val.txt",
                "nc: 1",
                "names: ['person']",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return dataset_yaml


def _resolve_yolov5_weights(weights: str, cfg: ExperimentConfig) -> str:
    if weights.strip():
        return weights
    return cfg.detector.yolov5_weights.strip() or "yolov5n.pt"


def _resolve_yolov5_device(cfg: ExperimentConfig) -> str:
    device = resolve_model_device(cfg.runtime.detector_device, cfg.runtime.device, "detector")
    if device.type == "cuda":
        return str(device.index if device.index is not None else 0)
    return "cpu"


def _read_yolov5_loss_history(results_csv: Path) -> list[float]:
    if not results_csv.exists():
        return []
    history: list[float] = []
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parts = []
            for key in ("train/box_loss", "train/obj_loss", "val/box_loss", "val/obj_loss"):
                if key in row and row[key]:
                    try:
                        parts.append(float(row[key]))
                    except ValueError:
                        continue
            if parts:
                history.append(sum(parts) / len(parts))
    return history


def train_yolov5_detector(cfg: ExperimentConfig, output_dir: str | Path, weights: str = "") -> dict[str, float | list[float]]:
    repo_dir = resolve_yolov5_repo_dir(cfg, workspace_hint=output_dir)
    train_script = repo_dir / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(f"YOLOv5 train.py was not found in '{repo_dir}'.")
    dataset_yaml = export_detector_dataset_to_yolov5(cfg, output_dir)
    run_root = ensure_dir(Path(output_dir) / "yolov5_runs")
    train_name = "train"
    cmd = [
        sys.executable,
        str(train_script),
        "--img",
        str(cfg.detector.yolov5_imgsz),
        "--batch",
        str(cfg.optim.batch_size),
        "--epochs",
        str(cfg.optim.epochs),
        "--data",
        str(dataset_yaml),
        "--weights",
        _resolve_yolov5_weights(weights, cfg),
        "--project",
        str(run_root),
        "--name",
        train_name,
        "--exist-ok",
        "--single-cls",
        "--device",
        _resolve_yolov5_device(cfg),
    ]
    start = time.perf_counter()
    try:
        subprocess.run(cmd, check=True, cwd=repo_dir)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Failed to launch YOLOv5 training. Python executable: {sys.executable}, repo: {repo_dir}, script: {train_script}"
        ) from exc
    elapsed = time.perf_counter() - start
    weights_dir = run_root / train_name / "weights"
    best_ckpt = weights_dir / "best.pt"
    last_ckpt = weights_dir / "last.pt"
    chosen = best_ckpt if best_ckpt.exists() else last_ckpt
    if not chosen.exists():
        raise FileNotFoundError(f"YOLOv5 training finished but no checkpoint was found in {weights_dir}")
    shutil.copy2(chosen, Path(output_dir) / "detector_last.pt")
    loss_history = _read_yolov5_loss_history(run_root / train_name / "results.csv")
    return {
        "loss_history": loss_history,
        "train_seconds_total": elapsed,
        "epoch_seconds_mean": elapsed / max(cfg.optim.epochs, 1),
        "train_peak_memory_mb": 0.0,
        "auto_resumed": False,
        "warm_started": bool(weights.strip()),
    }
