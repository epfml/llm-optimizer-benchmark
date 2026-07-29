#!/usr/bin/env python3
"""Wait for S2, select three challengers, and run paired-seed S3 confirmation."""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--train-data-path", type=Path, required=True)
    parser.add_argument("--val-data-path", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--lr-scale-file", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    return parser.parse_args()


def write_json_atomic(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def write_state(path, stage, **extra):
    write_json_atomic(
        path,
        {"stage": stage, "updated_at_unix": time.time(), **extra},
    )


def wait_for_main(state_path, poll_seconds):
    while True:
        if state_path.is_file():
            state = json.loads(state_path.read_text())
            stage = state.get("stage")
            if stage == "completed":
                return
            if stage == "failed":
                raise RuntimeError("S2 failed; S3 will not be started.")
        time.sleep(poll_seconds)


def select_challengers(results_dir):
    manifest_path = results_dir / "manifest_main.json"
    manifest = json.loads(manifest_path.read_text())
    scores = []
    for run in manifest.get("runs", []):
        if run.get("status") not in {"completed", "skipped-completed"}:
            continue
        summary_path = Path(run["experiment_dir"]) / "summary.json"
        summary = json.loads(summary_path.read_text())
        losses = summary.get("val_loss", [])
        if not losses:
            raise ValueError(f"No validation loss in {summary_path}")
        final_loss = float(losses[-1])
        if not math.isfinite(final_loss):
            raise ValueError(f"Non-finite final validation loss in {summary_path}")
        scores.append(
            {
                "optimizer": run["optimizer"],
                "final_val_loss": final_loss,
                "summary_path": str(summary_path),
            }
        )

    by_optimizer = {}
    for score in scores:
        previous = by_optimizer.get(score["optimizer"])
        if previous is None or score["final_val_loss"] < previous["final_val_loss"]:
            by_optimizer[score["optimizer"]] = score
    if len(by_optimizer) != 20:
        raise ValueError(
            f"Expected 20 completed S2 optimizers, found {len(by_optimizer)}."
        )

    ranking = sorted(
        by_optimizer.values(),
        key=lambda item: (item["final_val_loss"], item["optimizer"]),
    )
    challengers = [item["optimizer"] for item in ranking if item["optimizer"] != "adamw"][
        :3
    ]
    return ["adamw", *challengers], ranking


def run(command, cwd):
    print("$ " + " ".join(str(value) for value in command), flush=True)
    return subprocess.run(command, cwd=cwd, check=False).returncode


def manifest_counts(path):
    if not path.is_file():
        return {}
    manifest = json.loads(path.read_text())
    counts = {}
    for record in manifest.get("runs", []):
        status = record.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def main():
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.results_dir = args.results_dir.resolve()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.results_dir / "pipeline_state.json"

    try:
        wait_for_main(state_path, args.poll_seconds)
        write_state(state_path, "confirm_select")
        optimizers, ranking = select_challengers(args.results_dir)
        selection_path = args.results_dir / "confirm_selection.json"
        write_json_atomic(
            selection_path,
            {
                "selection_rule": (
                    "AdamW baseline plus the three non-AdamW optimizers with the "
                    "lowest S2 final validation loss at 268435456 tokens."
                ),
                "selected_optimizers": optimizers,
                "seeds": [1, 2],
                "ranking": ranking,
                "created_at_unix": time.time(),
            },
        )

        write_state(
            state_path,
            "confirm",
            selected_optimizers=optimizers,
            seeds=[1, 2],
            selection_file=str(selection_path),
        )
        returncode = run(
            [
                sys.executable,
                str(args.repo_root / "scripts/sub_one_pass/run_all.py"),
                "--stage",
                "confirm",
                "--optimizers",
                *optimizers,
                "--seeds",
                "1",
                "2",
                "--lr-scale-file",
                str(args.lr_scale_file),
                "--dataset",
                "token-bin",
                "--train-data-path",
                str(args.train_data_path),
                "--val-data-path",
                str(args.val_data_path),
                "--results-dir",
                str(args.results_dir),
            ],
            args.repo_root,
        )
        if returncode:
            write_state(
                state_path,
                "failed",
                failed_stage="confirm",
                returncode=returncode,
                selected_optimizers=optimizers,
            )
            return returncode

        write_state(
            state_path,
            "confirm_summarize",
            selected_optimizers=optimizers,
        )
        returncode = run(
            [
                sys.executable,
                str(args.repo_root / "scripts/sub_one_pass/summarize.py"),
                str(args.results_dir),
            ],
            args.repo_root,
        )
        if returncode:
            write_state(
                state_path,
                "failed",
                failed_stage="confirm_summarize",
                returncode=returncode,
                selected_optimizers=optimizers,
            )
            return returncode

        write_state(
            state_path,
            "completed",
            s3_completed=True,
            selected_optimizers=optimizers,
            seeds=[1, 2],
            main_manifest_counts=manifest_counts(
                args.results_dir / "manifest_main.json"
            ),
            confirm_manifest_counts=manifest_counts(
                args.results_dir / "manifest_confirm.json"
            ),
            selection_file=str(selection_path),
            summary_csv=str(args.results_dir / "sub_one_pass_results.csv"),
        )
        return 0
    except Exception as error:
        current_stage = None
        if state_path.is_file():
            current_stage = json.loads(state_path.read_text()).get("stage")
        if current_stage != "failed":
            write_state(
                state_path,
                "failed",
                failed_stage="confirm_gate",
                error=f"{type(error).__name__}: {error}",
            )
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
