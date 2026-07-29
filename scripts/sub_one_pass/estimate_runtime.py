#!/usr/bin/env python3
"""Estimate a serial single-GPU queue from completed calibration summaries."""

import argparse
import json
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", nargs="?", default="./exps/sub_one_pass")
    parser.add_argument("--target-tokens", type=int, default=268_435_456)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument(
        "--overhead-factor",
        type=float,
        default=1.15,
        help="Multiplier for evaluation, startup, checkpointing and queue overhead.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.target_tokens <= 0 or args.seeds <= 0 or args.overhead_factor < 1:
        raise ValueError("target-tokens and seeds must be positive; overhead >= 1.")

    throughputs = {}
    for path in Path(args.results_dir).rglob("summary.json"):
        payload = json.loads(path.read_text())
        optimizer = payload.get("args", {}).get("opt")
        throughput = payload.get("mean_tokens_per_second")
        if optimizer and throughput and throughput > 0:
            throughputs.setdefault(optimizer, []).append(float(throughput))

    if not throughputs:
        raise RuntimeError("No summaries with mean_tokens_per_second were found.")

    optimizer_tps = {
        optimizer: statistics.median(values)
        for optimizer, values in sorted(throughputs.items())
    }
    total_seconds = 0.0
    print("optimizer,tokens_per_second,estimated_hours")
    for optimizer, throughput in optimizer_tps.items():
        seconds = args.target_tokens * args.seeds / throughput * args.overhead_factor
        total_seconds += seconds
        print(f"{optimizer},{throughput:.1f},{seconds / 3600:.3f}")
    print(f"TOTAL,,{total_seconds / 3600:.3f}")
    print(
        "Estimate uses measured training throughput and a global overhead factor; "
        "it is not a billing guarantee."
    )


if __name__ == "__main__":
    main()
