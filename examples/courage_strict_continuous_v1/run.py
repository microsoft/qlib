"""Run the bounded continuous Courage Strict training experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from qlib.contrib.model.courage_strict_continuous_v1 import train_origin

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "examples/courage_strict_continuous_v1/config.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("smoke", "profile", "train-origin", "resume-origin")
    )
    parser.add_argument(
        "--origin",
        choices=("origin_2026_01", "origin_2026_02", "origin_2026_03"),
        default="origin_2026_01",
    )
    args = parser.parse_args()
    if args.command == "smoke":
        train_origin(
            CONFIG,
            origin=args.origin,
            output_override=ROOT / "artifacts/courage_strict_continuous_v1/smoke_v1",
            maximum_steps=2,
        )
    elif args.command == "profile":
        train_origin(
            CONFIG,
            origin=args.origin,
            output_override=ROOT
            / "artifacts/courage_strict_continuous_v1/profile_250_v1",
            maximum_steps=250,
        )
    elif args.command == "train-origin":
        train_origin(CONFIG, origin=args.origin)
    else:
        output = (
            ROOT / "artifacts/courage_strict_continuous_v1/run_v1" / args.origin
        )
        train_origin(
            CONFIG,
            origin=args.origin,
            resume_from=output / "last.pt",
        )


if __name__ == "__main__":
    main()
