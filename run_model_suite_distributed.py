from __future__ import annotations

import argparse

from lite_therm_pose.experiment_suite import run_suite
from lite_therm_pose.model_partition import assigned_models, partition_models
from lite_therm_pose.models.specs import MODEL_SPECS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a subset of the model suite assigned to one machine.")
    parser.add_argument("--pretrain-config", type=str, required=True)
    parser.add_argument("--finetune-config", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/model_suite")
    parser.add_argument("--machine-index", type=int, required=True)
    parser.add_argument("--num-machines", type=int, default=2)
    parser.add_argument("--models", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = [item.strip() for item in args.models.split(",") if item.strip()] if args.models else list(MODEL_SPECS)
    parts = partition_models(models, args.num_machines)
    for idx, part in enumerate(parts):
        print(f"machine_{idx}: {','.join(part)}")
    mine = assigned_models(models, args.machine_index, args.num_machines)
    print(f"running machine_{args.machine_index}: {','.join(mine)}")
    rows = run_suite(args.pretrain_config, args.finetune_config, mine, f"{args.output}/machine_{args.machine_index}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
