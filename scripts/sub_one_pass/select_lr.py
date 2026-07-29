#!/usr/bin/env python3
"""Select one learning-rate scale per optimizer from a tune-stage manifest."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest",
        nargs="?",
        default="./exps/sub_one_pass/manifest_tune.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Defaults to selected_lr_scales.json next to the manifest.",
    )
    return parser.parse_args()


def final_val_loss(summary_path):
    payload = json.loads(summary_path.read_text())
    history = payload.get("eval_history", [])
    if not history:
        raise ValueError(f"{summary_path} has no eval_history.")
    return float(max(history, key=lambda item: item["tokens"])["val_loss"])


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    candidates = {}
    missing = []

    for run in manifest.get("runs", []):
        summary_path = Path(run["experiment_dir"]) / "summary.json"
        if not summary_path.is_file():
            missing.append(str(summary_path))
            continue
        optimizer = run["optimizer"]
        candidate = {
            "lr_scale": float(run["lr_scale"]),
            "val_loss": final_val_loss(summary_path),
            "summary": str(summary_path),
        }
        candidates.setdefault(optimizer, []).append(candidate)

    if missing:
        raise RuntimeError(
            "Tune stage is incomplete; missing summary files:\n" + "\n".join(missing)
        )

    selected = {}
    report = {}
    for optimizer in manifest.get("optimizers", []):
        optimizer_candidates = sorted(
            candidates.get(optimizer, []), key=lambda item: item["lr_scale"]
        )
        if not optimizer_candidates:
            raise RuntimeError(f"No completed LR candidates for {optimizer}.")
        best = min(optimizer_candidates, key=lambda item: item["val_loss"])
        selected[optimizer] = best["lr_scale"]
        report[optimizer] = {
            "selected_lr_scale": best["lr_scale"],
            "selected_val_loss": best["val_loss"],
            "selection_on_grid_boundary": best
            in (optimizer_candidates[0], optimizer_candidates[-1]),
            "candidates": optimizer_candidates,
        }

    output_path = (
        Path(args.output)
        if args.output is not None
        else manifest_path.parent / "selected_lr_scales.json"
    )
    output_path.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
    report_path = output_path.with_name(output_path.stem + "_report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    boundary = [
        optimizer
        for optimizer, item in report.items()
        if item["selection_on_grid_boundary"]
    ]
    print(f"Wrote selected scales to {output_path}")
    print(f"Wrote selection report to {report_path}")
    if boundary:
        print(
            "Grid-boundary optima require an expanded LR candidate before main: "
            + ", ".join(boundary)
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
