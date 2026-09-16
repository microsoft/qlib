"""Bounded CUDA/BF16 runtime acceptance for Courage Strict V1."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from qlib.contrib.data.courage_strict_v1.dataset import CourageStrictV1Dataset
from qlib.contrib.model.patchtst_courage_strict_v1 import PatchTSTCourageStrictV1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qlib-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible to the V1 runtime profile")
    root = args.qlib_root.resolve()
    provider = root / "data/courage_strict_v1/qlib_provider"
    dataset = CourageStrictV1Dataset(
        provider_root=provider,
        segments={"train": ("2025-07-01", "2026-03-02")},
    ).prepare("train", cache_capacity=8)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )
    vocabulary = json.loads(
        (provider / "industry_vocabulary.json").read_text(encoding="utf-8")
    )
    device = torch.device("cuda", 0)
    model = PatchTSTCourageStrictV1(industry_vocab_size=len(vocabulary)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    processed = 0
    completed = 0
    for batch in loader:
        batch = {
            key: value.to(device, non_blocking=True) for key, value in batch.items()
        }
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(
                batch["values"],
                batch["available"],
                batch["missing"],
                batch["padding"],
                batch["minute_index"],
                batch["session_index"],
                batch["slow_values"],
                batch["slow_available"],
                batch["industry_id"],
            )
            errors = functional.huber_loss(
                prediction.float(), batch["targets"].float(), reduction="none"
            )
            denominator = batch["target_mask"].sum(dim=0).clamp_min(1)
            loss = ((errors * batch["target_mask"]).sum(dim=0) / denominator).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("runtime profile produced non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        completed += 1
        processed += len(batch["targets"])
        if completed >= args.steps:
            break
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    output = root / "artifacts/courage_strict_v1/runtime_profile"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "resume_probe.pt"
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint
    )
    restored = PatchTSTCourageStrictV1(industry_vocab_size=len(vocabulary)).to(device)
    restored.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
    report = {
        "schema_version": "courage_strict_v1_runtime_profile_v1",
        "decision": "PASS_V1_CUDA_BF16_RUNTIME_PROFILE",
        "device": torch.cuda.get_device_name(device),
        "batch_size": args.batch_size,
        "steps": completed,
        "samples": processed,
        "samples_per_second": processed / elapsed,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
        "finite_loss": float(loss.detach().cpu()),
        "checkpoint_resume_pass": True,
    }
    (output / "runtime_profile.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
