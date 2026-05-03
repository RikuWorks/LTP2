from __future__ import annotations

import argparse

from lite_therm_pose.experiment_suite import run_suite


LWIRPOSE_MODEL = "pose_resnet101_se_thermal_direct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain on COCO and fine-tune/evaluate on LWIRPOSE using S2-S7 train and S1 test."
    )
    parser.add_argument("--pretrain-config", type=str, default="configs/coco_pretrain_pose_direct.yaml")
    parser.add_argument("--finetune-config", type=str, default="configs/lwirpose_finetune_pose_direct.yaml")
    parser.add_argument("--output", type=str, default="outputs/lwirpose_pose_direct")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("lwirpose direct model:", LWIRPOSE_MODEL)
    rows = run_suite(args.pretrain_config, args.finetune_config, [LWIRPOSE_MODEL], args.output)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
