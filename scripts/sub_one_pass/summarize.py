#!/usr/bin/env python3
"""Collect sub-one-pass summary.json files into a tidy CSV."""

import argparse
import csv
import json
from pathlib import Path

FIELDS = [
    "optimizer",
    "seed",
    "data_seed",
    "tokens",
    "data_exposure",
    "iteration",
    "val_loss",
    "val_perplexity",
    "val_accuracy",
    "num_eval_batches",
    "train_time_seconds",
    "mean_tokens_per_second",
    "peak_memory_bytes",
    "experiment_dir",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", nargs="?", default="./exps/sub_one_pass")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def rows_from_summary(path):
    payload = json.loads(path.read_text())
    args = payload.get("args", {})
    for item in payload.get("eval_history", []):
        yield {
            "optimizer": args.get("opt"),
            "seed": args.get("seed"),
            "data_seed": args.get("data_seed"),
            "tokens": item.get("tokens"),
            "data_exposure": item.get("data_exposure"),
            "iteration": item.get("iter"),
            "val_loss": item.get("val_loss"),
            "val_perplexity": item.get("val_perplexity"),
            "val_accuracy": item.get("val_accuracy"),
            "num_eval_batches": item.get("num_eval_batches"),
            "train_time_seconds": payload.get("train_time_seconds"),
            "mean_tokens_per_second": payload.get("mean_tokens_per_second"),
            "peak_memory_bytes": payload.get("peak_memory_bytes"),
            "experiment_dir": str(path.parent),
        }


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output = (
        Path(args.output)
        if args.output is not None
        else results_dir / "sub_one_pass_results.csv"
    )
    rows = []
    for path in sorted(results_dir.rglob("summary.json")):
        rows.extend(rows_from_summary(path))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
