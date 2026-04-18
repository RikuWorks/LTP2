from __future__ import annotations

from lite_therm_pose.models.backbone import MODEL_SPECS


def main() -> None:
    for name, spec in MODEL_SPECS.items():
        print(f"{name}: block={spec.block_type}, width_mult={spec.width_mult}, decoder={spec.decoder_channels}")


if __name__ == "__main__":
    main()
