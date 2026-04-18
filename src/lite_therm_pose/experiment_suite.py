from __future__ import annotations

import json
import multiprocessing as mp
from copy import deepcopy
from pathlib import Path

from .config import ExperimentConfig, load_config
from .evaluate import evaluate_detector_model, evaluate_joint_pipeline, evaluate_pose_model, write_summary_report
from .profile import checkpoint_size_mb, parameter_stats
from .runtime import resolve_model_device
from .trainers import load_trained_detector, load_trained_pose, train_detector, train_pose
from .utils import ensure_dir


def clone_cfg(cfg: ExperimentConfig) -> ExperimentConfig:
    return deepcopy(cfg)


def _run_detector_job(cfg: ExperimentConfig, output_dir: str | Path, weights: str, queue: mp.Queue) -> None:
    try:
        _, stats = train_detector(cfg, output_dir, weights=weights)
        queue.put({"kind": "detector", "stats": stats, "error": ""})
    except Exception as exc:  # pragma: no cover
        queue.put({"kind": "detector", "stats": None, "error": repr(exc)})


def _run_pose_job(cfg: ExperimentConfig, output_dir: str | Path, weights: str, queue: mp.Queue) -> None:
    try:
        _, stats = train_pose(cfg, output_dir, weights=weights)
        queue.put({"kind": "pose", "stats": stats, "error": ""})
    except Exception as exc:  # pragma: no cover
        queue.put({"kind": "pose", "stats": None, "error": repr(exc)})


def _devices_allow_parallel(cfg: ExperimentConfig) -> bool:
    detector_device = resolve_model_device(cfg.runtime.detector_device, cfg.runtime.device, "detector")
    pose_device = resolve_model_device(cfg.runtime.pose_device, cfg.runtime.device, "pose")
    return detector_device.type == "cuda" and pose_device.type == "cuda" and detector_device != pose_device


def _train_phase_parallel(cfg: ExperimentConfig, det_output_dir: Path, pose_output_dir: Path, det_weights: str = "", pose_weights: str = "") -> tuple[dict[str, float | list[float]], dict[str, float | list[float]]]:
    if not _devices_allow_parallel(cfg):
        _, det_stats = train_detector(cfg, det_output_dir, weights=det_weights)
        _, pose_stats = train_pose(cfg, pose_output_dir, weights=pose_weights)
        return det_stats, pose_stats

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    det_proc = ctx.Process(target=_run_detector_job, args=(cfg, det_output_dir, det_weights, queue))
    pose_proc = ctx.Process(target=_run_pose_job, args=(cfg, pose_output_dir, pose_weights, queue))
    det_proc.start()
    pose_proc.start()

    results: dict[str, dict[str, float | list[float]]] = {}
    for _ in range(2):
        item = queue.get()
        if item["error"]:
            det_proc.join(timeout=1)
            pose_proc.join(timeout=1)
            raise RuntimeError(f"{item['kind']} training failed: {item['error']}")
        results[item["kind"]] = item["stats"]

    det_proc.join()
    pose_proc.join()
    if det_proc.exitcode != 0 or pose_proc.exitcode != 0:
        raise RuntimeError(f"parallel training failed: detector exit={det_proc.exitcode}, pose exit={pose_proc.exitcode}")
    return results["detector"], results["pose"]


def run_suite(
    pretrain_config_path: str,
    finetune_config_path: str,
    model_names: list[str],
    output_dir: str | Path,
) -> list[dict[str, float | str]]:
    pretrain_cfg_base = load_config(pretrain_config_path)
    finetune_cfg_base = load_config(finetune_config_path)
    root = ensure_dir(output_dir)
    summary_rows: list[dict[str, float | str]] = []

    for model_name in model_names:
        model_root = ensure_dir(root / model_name)
        pretrain_cfg = clone_cfg(pretrain_cfg_base)
        finetune_cfg = clone_cfg(finetune_cfg_base)
        pretrain_cfg.model.name = model_name
        finetune_cfg.model.name = model_name

        det_pre_dir = model_root / "pretrain_detector"
        pose_pre_dir = model_root / "pretrain_pose"
        det_fine_dir = model_root / "finetune_detector"
        pose_fine_dir = model_root / "finetune_pose"

        det_pre_stats, pose_pre_stats = _train_phase_parallel(pretrain_cfg, det_pre_dir, pose_pre_dir)
        det_fine_stats, pose_fine_stats = _train_phase_parallel(
            finetune_cfg,
            det_fine_dir,
            pose_fine_dir,
            det_weights=str(det_pre_dir / "detector_last.pt"),
            pose_weights=str(pose_pre_dir / "pose_last.pt"),
        )

        detector_fine = load_trained_detector(finetune_cfg, det_fine_dir / "detector_last.pt")
        pose_fine = load_trained_pose(finetune_cfg, pose_fine_dir / "pose_last.pt")

        det_metrics = evaluate_detector_model(detector_fine, finetune_cfg)
        pose_metrics = evaluate_pose_model(pose_fine, finetune_cfg)
        joint_metrics = evaluate_joint_pipeline(detector_fine, pose_fine, finetune_cfg)
        detector_profile = parameter_stats(detector_fine, "detector")
        pose_profile = parameter_stats(pose_fine, "pose")

        row: dict[str, float | str] = {
            "model": model_name,
            "model_family": finetune_cfg.model.name.split("_")[0],
            "det_pre_loss": round(float(det_pre_stats["loss_history"][-1]), 6) if det_pre_stats["loss_history"] else 0.0,
            "pose_pre_loss": round(float(pose_pre_stats["loss_history"][-1]), 6) if pose_pre_stats["loss_history"] else 0.0,
            "det_fine_loss": round(float(det_fine_stats["loss_history"][-1]), 6) if det_fine_stats["loss_history"] else 0.0,
            "pose_fine_loss": round(float(pose_fine_stats["loss_history"][-1]), 6) if pose_fine_stats["loss_history"] else 0.0,
            "det_pre_train_seconds": round(float(det_pre_stats["train_seconds_total"]), 6),
            "pose_pre_train_seconds": round(float(pose_pre_stats["train_seconds_total"]), 6),
            "det_fine_train_seconds": round(float(det_fine_stats["train_seconds_total"]), 6),
            "pose_fine_train_seconds": round(float(pose_fine_stats["train_seconds_total"]), 6),
            "det_epoch_seconds_mean": round(float(det_fine_stats["epoch_seconds_mean"]), 6),
            "pose_epoch_seconds_mean": round(float(pose_fine_stats["epoch_seconds_mean"]), 6),
            "det_train_peak_memory_mb": round(float(det_fine_stats["train_peak_memory_mb"]), 6),
            "pose_train_peak_memory_mb": round(float(pose_fine_stats["train_peak_memory_mb"]), 6),
            **{key: round(value, 6) for key, value in detector_profile.items()},
            **{key: round(value, 6) for key, value in pose_profile.items()},
            "total_params_m": round(detector_profile["detector_params_m"] + pose_profile["pose_params_m"], 6),
            "detector_checkpoint_mb": round(checkpoint_size_mb(det_fine_dir / "detector_last.pt"), 6),
            "pose_checkpoint_mb": round(checkpoint_size_mb(pose_fine_dir / "pose_last.pt"), 6),
            **{key: round(value, 6) for key, value in det_metrics.items()},
            **{key: round(value, 6) for key, value in pose_metrics.items()},
            **{key: round(value, 6) for key, value in joint_metrics.items()},
        }
        summary_rows.append(row)
        with (model_root / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(row, handle, ensure_ascii=False, indent=2)

    write_summary_report(summary_rows, root)
    return summary_rows
