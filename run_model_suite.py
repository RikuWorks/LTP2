from __future__ import annotations

import argparse

from lite_therm_pose.experiment_suite import run_suite
from lite_therm_pose.models.backbone import MODEL_SPECS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and benchmark all models with pretrain->finetune->test.")
    parser.add_argument("--pretrain-config", type=str, required=True)
    parser.add_argument("--finetune-config", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/model_suite")
    parser.add_argument("--models", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = [item.strip() for item in args.models.split(",") if item.strip()] if args.models else list(MODEL_SPECS)
    rows = run_suite(args.pretrain_config, args.finetune_config, models, args.output)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
