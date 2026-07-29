#!/usr/bin/env python3
"""Resume a server experiment at the main stage after manual LR acceptance."""

import argparse
import json
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
    parser.add_argument(
        "--accepted-boundary-optimizer",
        action="append",
        default=[],
    )
    return parser.parse_args()


def write_state(path, stage, **extra):
    payload = {"stage": stage, "updated_at_unix": time.time(), **extra}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def run(command, cwd):
    print("$ " + " ".join(str(value) for value in command), flush=True)
    return subprocess.run(command, cwd=cwd, check=False).returncode


def main():
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.results_dir / "pipeline_state.json"
    write_state(
        state_path,
        "main",
        lr_scale_file=str(args.lr_scale_file),
        accepted_boundary_optimizers=args.accepted_boundary_optimizer,
    )
    returncode = run(
        [
            sys.executable,
            str(args.repo_root / "scripts/sub_one_pass/run_all.py"),
            "--stage",
            "main",
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
        write_state(state_path, "failed", returncode=returncode)
        return returncode

    write_state(state_path, "summarize")
    returncode = run(
        [
            sys.executable,
            str(args.repo_root / "scripts/sub_one_pass/summarize.py"),
            str(args.results_dir),
        ],
        args.repo_root,
    )
    if returncode:
        write_state(state_path, "failed", returncode=returncode)
        return returncode
    write_state(state_path, "completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
