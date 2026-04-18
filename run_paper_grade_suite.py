from __future__ import annotations

import argparse

from lite_therm_pose.experiment_suite import run_suite
from lite_therm_pose.models.specs import PAPER_GRADE_MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 5 high-accuracy paper-grade model suite.")
    parser.add_argument("--pretrain-config", type=str, required=True)
    parser.add_argument("--finetune-config", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/paper_grade_suite")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("paper-grade models:", ",".join(PAPER_GRADE_MODELS))
    rows = run_suite(args.pretrain_config, args.finetune_config, PAPER_GRADE_MODELS, args.output)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
