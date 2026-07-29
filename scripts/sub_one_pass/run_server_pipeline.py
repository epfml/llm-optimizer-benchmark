#!/usr/bin/env python3
"""Run tune and main stages on a server after tokenization completes."""

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
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-pid", type=int, default=None)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--data-timeout-hours", type=float, default=6.0)
    parser.add_argument("--max-lr-expansions", type=int, default=2)
    return parser.parse_args()


def process_alive(pid):
    try:
        Path(f"/proc/{pid}").stat()
        return True
    except FileNotFoundError:
        return False


def write_state(path, stage, **extra):
    payload = {"stage": stage, "updated_at_unix": time.time(), **extra}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def wait_for_data(args, state_path):
    deadline = time.monotonic() + args.data_timeout_hours * 3600
    while time.monotonic() < deadline:
        if (
            args.data_manifest.is_file()
            and args.train_data_path.is_file()
            and args.val_data_path.is_file()
        ):
            return
        if args.tokenizer_pid is not None and not process_alive(args.tokenizer_pid):
            raise RuntimeError(
                "Tokenizer exited before producing the data manifest. "
                "Inspect the tokenization log; the pipeline will not retry it."
            )
        write_state(state_path, "waiting-for-data")
        time.sleep(args.poll_seconds)
    raise TimeoutError("Timed out waiting for tokenized data.")


def run(command, cwd):
    print("$ " + " ".join(str(value) for value in command), flush=True)
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"Command exited with status {completed.returncode}; not retrying."
        )


def common_runner_args(args):
    return [
        "--dataset",
        "token-bin",
        "--train-data-path",
        str(args.train_data_path),
        "--val-data-path",
        str(args.val_data_path),
        "--results-dir",
        str(args.results_dir),
    ]


def select_and_expand_lr(args, state_path):
    selection_script = args.repo_root / "scripts/sub_one_pass/select_lr.py"
    runner_script = args.repo_root / "scripts/sub_one_pass/run_all.py"
    tune_manifest = args.results_dir / "manifest_tune.json"
    selected_path = args.results_dir / "selected_lr_scales.json"
    report_path = args.results_dir / "selected_lr_scales_report.json"

    for expansion_round in range(args.max_lr_expansions + 1):
        completed = subprocess.run(
            [
                sys.executable,
                str(selection_script),
                str(tune_manifest),
                "--output",
                str(selected_path),
            ],
            cwd=args.repo_root,
            check=False,
        )
        if completed.returncode == 0:
            return selected_path
        if completed.returncode != 2:
            raise RuntimeError(
                f"LR selection exited with status {completed.returncode}."
            )
        if expansion_round == args.max_lr_expansions:
            raise RuntimeError(
                "LR optimum remains on a grid boundary after the allowed "
                "expansions. Stopping before main."
            )

        report = json.loads(report_path.read_text())
        boundary_optimizers = [
            optimizer
            for optimizer, item in report.items()
            if item["selection_on_grid_boundary"]
        ]
        write_state(
            state_path,
            "expanding-lr-grid",
            expansion_round=expansion_round + 1,
            optimizers=boundary_optimizers,
        )
        for optimizer in boundary_optimizers:
            candidates = report[optimizer]["candidates"]
            selected = report[optimizer]["selected_lr_scale"]
            scales = sorted(item["lr_scale"] for item in candidates)
            new_scale = selected / 3 if selected == scales[0] else selected * 3
            run(
                [
                    sys.executable,
                    str(runner_script),
                    "--stage",
                    "tune",
                    "--optimizers",
                    optimizer,
                    "--lr-scales",
                    str(new_scale),
                    *common_runner_args(args),
                ],
                args.repo_root,
            )
    raise AssertionError("Unreachable")


def main():
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.results_dir / "pipeline_state.json"
    runner_script = args.repo_root / "scripts/sub_one_pass/run_all.py"

    write_state(state_path, "waiting-for-data")
    wait_for_data(args, state_path)
    data_manifest = json.loads(args.data_manifest.read_text())
    if data_manifest.get("train_tokens", 0) <= 268_435_456:
        raise ValueError("Formal train corpus must contain more than 256M tokens.")

    write_state(state_path, "tune")
    run(
        [
            sys.executable,
            str(runner_script),
            "--stage",
            "tune",
            *common_runner_args(args),
        ],
        args.repo_root,
    )

    selected_path = select_and_expand_lr(args, state_path)
    write_state(state_path, "main", lr_scale_file=str(selected_path))
    run(
        [
            sys.executable,
            str(runner_script),
            "--stage",
            "main",
            "--lr-scale-file",
            str(selected_path),
            *common_runner_args(args),
        ],
        args.repo_root,
    )

    write_state(state_path, "summarize")
    run(
        [
            sys.executable,
            str(args.repo_root / "scripts/sub_one_pass/summarize.py"),
            str(args.results_dir),
        ],
        args.repo_root,
    )
    write_state(state_path, "completed")


if __name__ == "__main__":
    main()
