from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CORE_ABLATION_MODELS = [
    "pose_resnet50",
    "pose_resnet101",
    "pose_resnet50_se",
    "pose_resnet101_se",
    "pose_resnet50_se_thermal",
    "pose_resnet101_se_thermal",
    "pose_resnet101_se_thermal_ppm",
]

DIRECT_COMPARISON_MODELS = [
    "pose_resnet50_se_direct",
    "pose_resnet50_se_direct_refine",
    "pose_resnet101_se_direct",
    "pose_resnet50_se_thermal_direct",
    "pose_resnet101_se_thermal_direct",
]

LEGACY_REFERENCE_MODELS = [
    "hrnet_w48",
    "higherhrnet_w32",
    "thermhr_csp_bifpn",
    "stacked_hourglass_large",
]

KNOWN_NEGATIVE_MODELS = [
    "pose_resnet101_se_fuse",
    "pose_resnet101_se_thermal_ppm_xh",
    "pose_resnet101_se_thermal_elite_ppm",
    "pose_resnet101_se_thermal_dualpath_ppm",
    "pose_resnet101_se_thermal_dualpath_refine_ppm",
    "pose_resnet101_se_thermal_dualpath_modulated_ppm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a paper-oriented manifest from existing training and benchmark outputs.")
    parser.add_argument("--outputs-root", type=str, default="outputs")
    parser.add_argument("--benchmark-summary", type=str, default="outputs/otp2_test_benchmark2/summary.csv")
    parser.add_argument("--output", type=str, default="outputs/paper_model_selection")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def float_or_zero(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def model_key(row: dict[str, str]) -> str:
    return row.get("model_name") or row.get("model") or row.get("run_id") or ""


def architecture_family(model_name: str) -> str:
    if "direct" in model_name:
        return "direct"
    return "heatmap"


def merge_training_summaries(outputs_root: Path) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for summary_path in sorted(outputs_root.rglob("summary.csv")):
        if summary_path.parent.name.startswith("otp2_test_benchmark"):
            continue
        for row in read_csv_rows(summary_path):
            key = model_key(row)
            if not key:
                continue
            merged.setdefault(key, {}).update(row)
            merged[key]["training_summary_path"] = str(summary_path)
    return merged


def merge_benchmark_summary(summary_path: Path) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in read_csv_rows(summary_path):
        key = model_key(row)
        if not key:
            continue
        existing = merged.get(key)
        if existing is None or float_or_zero(row.get("joint_score_test")) > float_or_zero(existing.get("joint_score_test")):
            merged[key] = dict(row)
            merged[key]["benchmark_summary_path"] = str(summary_path)
    return merged


def combine_records(training_rows: dict[str, dict[str, Any]], benchmark_rows: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    keys = set(training_rows) | set(benchmark_rows)
    for key in keys:
        combined: dict[str, Any] = {"model": key}
        combined.update(training_rows.get(key, {}))
        combined.update(benchmark_rows.get(key, {}))
        combined["architecture"] = architecture_family(key)
        combined["joint_score_sort"] = float_or_zero(combined.get("joint_score_test") or combined.get("joint_score"))
        combined["pose_pckh50_sort"] = float_or_zero(combined.get("pose_pckh50_test") or combined.get("pose_pckh50"))
        combined["gpu_joint_fps_sort"] = float_or_zero(combined.get("gpu_joint_fps_images") or combined.get("joint_fps"))
        combined["total_params_sort"] = float_or_zero(combined.get("total_params_m"))
        records[key] = combined
    return records


def best_available(records: dict[str, dict[str, Any]], model_names: list[str]) -> list[dict[str, Any]]:
    selected = [records[name] for name in model_names if name in records]
    return selected


def top_by(records: dict[str, dict[str, Any]], predicate, key_name: str, limit: int = 1) -> list[dict[str, Any]]:
    filtered = [row for row in records.values() if predicate(row)]
    filtered.sort(key=lambda row: row.get(key_name, 0.0), reverse=True)
    return filtered[:limit]


def paper_selection(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    main_heatmap = top_by(records, lambda row: row["architecture"] == "heatmap", "joint_score_sort", limit=2)
    main_direct = top_by(records, lambda row: row["architecture"] == "direct", "joint_score_sort", limit=3)
    lightweight = top_by(
        records,
        lambda row: row["total_params_sort"] > 0.0 and row["joint_score_sort"] >= 0.75,
        "gpu_joint_fps_sort",
        limit=3,
    )
    ablation = best_available(records, CORE_ABLATION_MODELS)
    direct_family = best_available(records, DIRECT_COMPARISON_MODELS)
    legacy = best_available(records, LEGACY_REFERENCE_MODELS)
    negative = best_available(records, KNOWN_NEGATIVE_MODELS)

    rerun_with_yolo = []
    seen: set[str] = set()
    for group in (ablation, direct_family, legacy):
        for row in group:
            model_name = row["model"]
            if model_name in seen:
                continue
            seen.add(model_name)
            rerun_with_yolo.append(model_name)

    return {
        "main_table": [row["model"] for row in main_heatmap + main_direct],
        "speed_table": [row["model"] for row in lightweight],
        "ablation_table": [row["model"] for row in ablation],
        "direct_table": [row["model"] for row in direct_family],
        "legacy_reference_table": [row["model"] for row in legacy],
        "negative_results_appendix": [row["model"] for row in negative],
        "rerun_with_yolo": rerun_with_yolo,
    }


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    columns = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{col: row.get(col, "") for col in columns} for row in rows])


def write_markdown(records: dict[str, dict[str, Any]], selection: dict[str, Any], output_dir: Path) -> None:
    sections = [
        ("Main Table", selection["main_table"]),
        ("Speed Table", selection["speed_table"]),
        ("Ablation Table", selection["ablation_table"]),
        ("Direct Table", selection["direct_table"]),
        ("Legacy Reference Table", selection["legacy_reference_table"]),
        ("Negative Results Appendix", selection["negative_results_appendix"]),
        ("Rerun With YOLO", selection["rerun_with_yolo"]),
    ]
    lines = ["# Paper Model Selection", ""]
    for title, model_names in sections:
        lines.append(f"## {title}")
        if not model_names:
            lines.append("")
            lines.append("No matching models were found in the current outputs.")
            lines.append("")
            continue
        lines.append("")
        lines.append("| model | architecture | joint_score | pose_pckh50 | gpu_joint_fps | total_params_m | source |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for model_name in model_names:
            row = records.get(model_name, {"model": model_name})
            source = row.get("benchmark_summary_path") or row.get("training_summary_path") or ""
            lines.append(
                "| "
                + " | ".join(
                    [
                        model_name,
                        str(row.get("architecture", "")),
                        str(row.get("joint_score_test") or row.get("joint_score") or ""),
                        str(row.get("pose_pckh50_test") or row.get("pose_pckh50") or ""),
                        str(row.get("gpu_joint_fps_images") or row.get("joint_fps") or ""),
                        str(row.get("total_params_m") or ""),
                        str(source),
                    ]
                )
                + " |"
            )
        lines.append("")
    (output_dir / "paper_model_selection.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_rows = merge_training_summaries(outputs_root)
    benchmark_rows = merge_benchmark_summary(Path(args.benchmark_summary))
    records = combine_records(training_rows, benchmark_rows)
    selection = paper_selection(records)

    with (output_dir / "recommended_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, ensure_ascii=False, indent=2)

    merged_rows = sorted(records.values(), key=lambda row: (row["joint_score_sort"], row["pose_pckh50_sort"]), reverse=True)
    write_rows(merged_rows, output_dir / "all_models_merged.csv")
    write_markdown(records, selection, output_dir)


if __name__ == "__main__":
    main()
