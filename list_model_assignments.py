from __future__ import annotations

import argparse

from lite_therm_pose.model_partition import partition_models
from lite_therm_pose.models.specs import MODEL_SPECS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show how models are split across machines.")
    parser.add_argument("--num-machines", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for idx, models in enumerate(partition_models(list(MODEL_SPECS), args.num_machines)):
        print(f"machine_{idx} ({len(models)} models)")
        for model in models:
            print(f"  - {model}")


if __name__ == "__main__":
    main()
