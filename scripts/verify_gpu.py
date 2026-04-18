from __future__ import annotations

import os
import sys

import torch


def main() -> int:
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"cuda_version: {torch.version.cuda}")
    print(f"device_count: {torch.cuda.device_count()}")
    print(f"cuda_visible_devices: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            print(f"device_{idx}: {torch.cuda.get_device_name(idx)}")
        return 0
    print("device_name: none")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
