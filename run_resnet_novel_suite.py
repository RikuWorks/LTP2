from __future__ import annotations

import argparse

from lite_therm_pose.experiment_suite import run_suite
from lite_therm_pose.models.specs import RESNET_NOVEL_MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the novel thermal-aware ResNet suite.")
    parser.add_argument("--pretrain-config", type=str, required=True)
    parser.add_argument("--finetune-config", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/resnet_novel_suite")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("resnet novel models:", ",".join(RESNET_NOVEL_MODELS))
    rows = run_suite(args.pretrain_config, args.finetune_config, RESNET_NOVEL_MODELS, args.output)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
