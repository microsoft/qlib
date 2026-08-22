"""Run the original Continuous V1 population with step-level diagnostics."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from qlib.contrib.model.courage_strict_continuous_v1 import sha256_file, train_origin

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "examples/courage_strict_continuous_v1/config_diagnostic_v1.json"
CONTRACT = ROOT / "examples/courage_strict_continuous_v1/diagnostic_contract_v1.json"
OUTPUT = (
    ROOT / "artifacts/courage_strict_continuous_v1_diagnostic_v1/run_v1/origin_2026_01"
)


def validate_contract() -> dict:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if (
        contract.get("schema_version")
        != "courage_strict_continuous_diagnostic_contract_v1"
        or contract.get("experiment_id") != "courage_strict_continuous_v1_diagnostic_v1"
        or contract["scope"].get("april_or_later_read") is not False
    ):
        raise RuntimeError("diagnostic contract identity drift")
    for key, hash_key in (
        ("source_config", "source_config_sha256"),
        ("diagnostic_config", "diagnostic_config_sha256"),
        ("trainer", "trainer_sha256"),
        ("runner", "runner_sha256"),
    ):
        path = ROOT / contract[key]
        if not path.is_file() or sha256_file(path) != contract[hash_key]:
            raise RuntimeError(f"diagnostic frozen input drift: {key}")
    source = json.loads((ROOT / contract["source_config"]).read_text(encoding="utf-8"))
    diagnostic = json.loads(CONFIG.read_text(encoding="utf-8"))
    normalized = copy.deepcopy(diagnostic)
    normalized["experiment_id"] = source["experiment_id"]
    normalized["roots"]["output_root"] = source["roots"]["output_root"]
    normalized.pop("diagnostics")
    if normalized != source:
        raise RuntimeError("diagnostic experiment changes more than logging identity")
    forbidden = set(contract["authority"]) - {
        "training",
        "checkpoint_write",
        "valid_evaluation",
    }
    if any(contract["authority"][name] is not False for name in forbidden):
        raise RuntimeError("diagnostic forbidden authority drift")
    return contract


def main() -> None:
    validate_contract()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "train", "resume"))
    args = parser.parse_args()
    if args.command == "smoke":
        train_origin(
            CONFIG,
            origin="origin_2026_01",
            output_override=ROOT
            / "artifacts/courage_strict_continuous_v1_diagnostic_v1/smoke_v1",
            maximum_steps=2,
        )
    elif args.command == "train":
        train_origin(CONFIG, origin="origin_2026_01")
    else:
        train_origin(
            CONFIG,
            origin="origin_2026_01",
            resume_from=OUTPUT / "last.pt",
        )


if __name__ == "__main__":
    main()
