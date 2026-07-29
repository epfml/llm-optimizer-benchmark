#!/usr/bin/env python3
"""Run every optimizer through one nested, single-GPU token-budget trajectory."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

STAGE_EVAL_TOKENS = {
    "smoke": [2_097_152],
    "tune": [16_777_216],
    "main": [
        16_777_216,
        33_554_432,
        67_108_864,
        134_217_728,
        268_435_456,
    ],
    "confirm": [
        16_777_216,
        33_554_432,
        67_108_864,
        134_217_728,
        268_435_456,
    ],
}

OPTIMIZER_ARGS = {
    "adamw": ["--lr", "1e-3", "--beta1", "0.8", "--beta2", "0.999"],
    "sgd": ["--lr", "1e-2", "--momentum", "0.9"],
    "muon": [
        "--lr",
        "1e-3",
        "--muon_lr_factor",
        "1e-2",
        "--beta1",
        "0.8",
        "--beta2",
        "0.999",
        "--momentum",
        "0.95",
        "--nesterov",
        "True",
    ],
    "soap": ["--lr", "1e-3", "--beta1", "0.9", "--beta2", "0.999"],
    "ademamix": [
        "--lr",
        "1e-3",
        "--beta1",
        "0.9",
        "--beta2",
        "0.999",
        "--adema_beta3",
        "0.999",
        "--adema_alpha",
        "8.0",
    ],
    "lion": ["--lr", "1e-3", "--beta1", "0.9", "--beta2", "0.99"],
    "sf-adamw": [
        "--lr",
        "1e-3",
        "--beta1",
        "0.9",
        "--beta2",
        "0.9999",
        "--scheduler",
        "none",
    ],
    "sf-sgd": [
        "--lr",
        "1e-2",
        "--momentum",
        "0.9",
        "--scheduler",
        "none",
    ],
    "signsgd": ["--lr", "1e-3"],
    "signum": ["--lr", "1e-3", "--momentum", "0.9"],
    "prodigy": [
        "--lr",
        "1.0",
        "--beta1",
        "0.9",
        "--beta2",
        "0.999",
        "--prodigy_use_bias_correction",
        "True",
    ],
    "sophiag": ["--lr", "1e-3", "--beta1", "0.9", "--beta2", "0.999"],
    "adopt": ["--lr", "1e-3", "--beta1", "0.9", "--beta2", "0.999"],
    "mars": [
        "--lr",
        "1e-3",
        "--mars_lr",
        "3e-3",
        "--beta1",
        "0.8",
        "--beta2",
        "0.999",
        "--mars_beta1",
        "0.95",
        "--mars_beta2",
        "0.99",
    ],
    "adafactor": ["--lr", "1e-3", "--beta1", "0.9"],
    "lamb": ["--lr", "1e-3", "--beta1", "0.9", "--beta2", "0.999"],
    "scion": ["--lr", "1e-3", "--momentum", "0.9"],
    "scion-light": ["--lr", "1e-3", "--momentum", "0.9"],
    "d-muon": [
        "--lr",
        "1e-3",
        "--beta1",
        "0.8",
        "--beta2",
        "0.999",
        "--momentum",
        "0.95",
        "--nesterov",
        "True",
    ],
    "muon-pytorch": [
        "--lr",
        "2e-2",
        "--muon_adamw_lr",
        "1e-3",
        "--momentum",
        "0.95",
        "--nesterov",
        "True",
    ],
}

LR_FLAGS = {
    "adamw": {"--lr"},
    "sgd": {"--lr"},
    "muon": {"--lr", "--muon_lr_factor"},
    "soap": {"--lr"},
    "ademamix": {"--lr"},
    "lion": {"--lr"},
    "sf-adamw": {"--lr"},
    "sf-sgd": {"--lr"},
    "signsgd": {"--lr"},
    "signum": {"--lr"},
    "prodigy": {"--lr"},
    "sophiag": {"--lr"},
    "adopt": {"--lr"},
    "mars": {"--lr", "--mars_lr"},
    "adafactor": {"--lr"},
    "lamb": {"--lr"},
    "scion": {"--lr"},
    "scion-light": {"--lr"},
    "d-muon": {"--lr"},
    "muon-pytorch": {"--lr", "--muon_adamw_lr"},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sequentially benchmark all repository optimizers on one GPU."
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=["all"],
        help="Optimizer names or 'all'.",
    )
    parser.add_argument("--dataset", default="fineweb")
    parser.add_argument("--datasets-dir", default="./src/data/datasets/")
    parser.add_argument("--train-data-path", default=None)
    parser.add_argument("--val-data-path", default=None)
    parser.add_argument("--results-dir", default="./exps/sub_one_pass")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Run multiple model seeds. Overrides --seed.",
    )
    parser.add_argument("--data-seed", type=int, default=1337)
    parser.add_argument(
        "--stage",
        choices=sorted(STAGE_EVAL_TOKENS),
        default="main",
        help="Use the default token budget and LR grid for this protocol stage.",
    )
    parser.add_argument(
        "--eval-tokens",
        type=int,
        nargs="+",
        default=None,
        help="Override the selected stage's token checkpoints.",
    )
    parser.add_argument(
        "--protocol-horizon-tokens",
        type=int,
        default=268_435_456,
        help="Fixed horizon for optimizer-internal schedules such as AdEMAMix.",
    )
    parser.add_argument(
        "--lr-scales",
        type=float,
        nargs="+",
        default=None,
        help="Scale all learning-rate fields. Tune defaults to 0.3, 1, 3.",
    )
    parser.add_argument(
        "--lr-scale-file",
        default=None,
        help="JSON object mapping optimizer names to selected LR scales.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--acc-steps", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="llm-optimizer-sub-one-pass")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Run even when the experiment directory already has summary.json.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.eval_tokens is None:
        args.eval_tokens = list(STAGE_EVAL_TOKENS[args.stage])
    if args.seeds is None:
        args.seeds = [args.seed]
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("--seeds must be unique.")
    if args.lr_scales is None:
        args.lr_scales = [0.3, 1.0, 3.0] if args.stage == "tune" else [1.0]
    if any(value <= 0 for value in args.lr_scales):
        raise ValueError("--lr-scales values must be positive.")
    if len(args.lr_scales) != len(set(args.lr_scales)):
        raise ValueError("--lr-scales must be unique.")

    unknown = set(args.optimizers) - set(OPTIMIZER_ARGS) - {"all"}
    if unknown:
        raise ValueError(f"Unknown optimizers: {sorted(unknown)}")
    if "all" in args.optimizers and len(args.optimizers) != 1:
        raise ValueError("'all' cannot be combined with explicit optimizer names.")
    if not args.eval_tokens:
        raise ValueError("--eval-tokens must contain at least one token boundary.")
    if any(value <= 0 for value in args.eval_tokens):
        raise ValueError("--eval-tokens values must be positive.")
    if args.eval_tokens != sorted(set(args.eval_tokens)):
        raise ValueError("--eval-tokens must be strictly increasing and unique.")

    tokens_per_step = args.batch_size * args.acc_steps * args.sequence_length
    invalid = [value for value in args.eval_tokens if value % tokens_per_step]
    if invalid:
        raise ValueError(
            f"Token boundaries {invalid} are not divisible by "
            f"tokens_per_step={tokens_per_step}."
        )
    if args.protocol_horizon_tokens < args.eval_tokens[-1]:
        raise ValueError("--protocol-horizon-tokens must cover the largest checkpoint.")
    if args.protocol_horizon_tokens % tokens_per_step:
        raise ValueError(
            "--protocol-horizon-tokens must be divisible by "
            f"tokens_per_step={tokens_per_step}."
        )
    if args.n_embd % args.n_head:
        raise ValueError("--n-embd must be divisible by --n-head.")
    if args.dataset == "token-bin" and not (
        args.train_data_path and args.val_data_path
    ):
        raise ValueError(
            "--dataset token-bin requires --train-data-path and --val-data-path."
        )
    if args.lr_scale_file is not None and len(args.lr_scales) != 1:
        raise ValueError(
            "--lr-scale-file cannot be combined with multiple --lr-scales."
        )


def load_lr_scale_map(path):
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("--lr-scale-file must contain a JSON object.")
    unknown = set(payload) - set(OPTIMIZER_ARGS)
    if unknown:
        raise ValueError(f"Unknown optimizers in LR scale file: {sorted(unknown)}")
    scales = {}
    for optimizer, value in payload.items():
        value = float(value)
        if value <= 0:
            raise ValueError(f"LR scale for {optimizer} must be positive.")
        scales[optimizer] = value
    return scales


def scaled_optimizer_args(optimizer, scale):
    arguments = list(OPTIMIZER_ARGS[optimizer])
    lr_flags = LR_FLAGS[optimizer]
    for index in range(0, len(arguments), 2):
        if arguments[index] in lr_flags:
            arguments[index + 1] = f"{float(arguments[index + 1]) * scale:.12g}"
    return arguments


def format_scale(scale):
    return f"{scale:.6g}".replace("-", "m").replace(".", "p")


def command_for(args, optimizer, repo_root, seed=None, lr_scale=1.0):
    seed = args.seed if seed is None else seed
    max_tokens = args.eval_tokens[-1]
    steps = max_tokens // (args.batch_size * args.acc_steps * args.sequence_length)
    protocol_horizon_steps = args.protocol_horizon_tokens // (
        args.batch_size * args.acc_steps * args.sequence_length
    )
    experiment_name = (
        f"sub1_{args.stage}_{args.dataset}_{optimizer}_seed{seed}_"
        f"lrscale{format_scale(lr_scale)}_tokens{max_tokens}"
    )

    if optimizer == "d-muon":
        environment_torchrun = Path(sys.executable).with_name("torchrun")
        torchrun = (
            str(environment_torchrun)
            if environment_torchrun.is_file()
            else shutil.which("torchrun")
        )
        if torchrun is None:
            raise RuntimeError("d-muon requires torchrun, but it is not on PATH.")
        command = [
            torchrun,
            "--standalone",
            "--nproc_per_node=1",
            str(repo_root / "src" / "main.py"),
            "--distributed_backend",
            "nccl",
        ]
    else:
        command = [sys.executable, str(repo_root / "src" / "main.py")]

    command += [
        "--config_format",
        "base",
        "--model",
        "llama",
        "--dataset",
        args.dataset,
        "--datasets_dir",
        args.datasets_dir,
        "--device",
        args.device,
        "--n_layer",
        str(args.n_layer),
        "--n_head",
        str(args.n_head),
        "--n_embd",
        str(args.n_embd),
        "--batch_size",
        str(args.batch_size),
        "--acc_steps",
        str(args.acc_steps),
        "--sequence_length",
        str(args.sequence_length),
        "--opt",
        optimizer,
        "--train_token_budget",
        str(max_tokens),
        "--eval_at_tokens",
        *[str(value) for value in args.eval_tokens],
        "--strict_sub_one_pass",
        "--fixed_data_boundaries",
        "--lazy_data_permutation",
        "--scheduler",
        "warmup_constant",
        "--warmup_steps",
        str(args.warmup_steps),
        "--eval_interval",
        "0",
        "--eval_batches",
        str(args.eval_batches),
        "--limit_final_eval",
        "--latest_ckpt_interval",
        str(steps),
        "--log_interval",
        "50",
        "--weight_decay",
        str(args.weight_decay),
        "--grad_clip",
        str(args.grad_clip),
        "--seed",
        str(seed),
        "--data_seed",
        str(args.data_seed),
        "--results_base_folder",
        args.results_dir,
        "--experiment_name",
        experiment_name,
    ]
    command += scaled_optimizer_args(optimizer, lr_scale)
    if args.dataset == "token-bin":
        command += [
            "--train_data_path",
            args.train_data_path,
            "--val_data_path",
            args.val_data_path,
        ]

    # Keep short stages prefix-compatible with the main trajectory.
    if optimizer == "ademamix":
        command += [
            "--adema_beta3_warmup",
            str(protocol_horizon_steps),
            "--adema_alpha_warmup",
            str(protocol_horizon_steps),
        ]

    if args.wandb:
        command += ["--wandb", "--wandb_project", args.wandb_project]
        if args.wandb_entity:
            command += ["--wandb_entity", args.wandb_entity]
    return command


def write_manifest(path, manifest):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary_path.replace(path)


def initialize_manifest(path, args, optimizers):
    invocation = {"arguments": vars(args), "started_at_unix": time.time()}
    if path.is_file():
        manifest = json.loads(path.read_text())
        if (
            manifest.get("protocol") != "sub-one-pass"
            or manifest.get("stage") != args.stage
        ):
            raise ValueError(f"{path} belongs to a different experiment protocol.")
        manifest["optimizers"] = list(
            dict.fromkeys(manifest.get("optimizers", []) + optimizers)
        )
        manifest.setdefault("invocations", []).append(invocation)
        manifest.setdefault("runs", [])
        return manifest
    return {
        "protocol": "sub-one-pass",
        "stage": args.stage,
        "optimizers": optimizers,
        "invocations": [invocation],
        "runs": [],
    }


def upsert_run_record(manifest, run_record):
    for existing in manifest["runs"]:
        if existing.get("experiment_dir") == run_record["experiment_dir"]:
            existing.clear()
            existing.update(run_record)
            return existing
    manifest["runs"].append(run_record)
    return run_record


def run_and_tee(command, repo_root, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
        return process.wait()


def normalize_paths(args):
    args.results_dir = str(Path(args.results_dir).resolve())
    args.datasets_dir = str(Path(args.datasets_dir).resolve())
    if args.train_data_path is not None:
        args.train_data_path = str(Path(args.train_data_path).resolve())
    if args.val_data_path is not None:
        args.val_data_path = str(Path(args.val_data_path).resolve())
    if args.lr_scale_file is not None:
        args.lr_scale_file = str(Path(args.lr_scale_file).resolve())


def main():
    args = parse_args()
    validate_args(args)
    normalize_paths(args)
    repo_root = Path(__file__).resolve().parents[2]
    optimizers = list(OPTIMIZER_ARGS) if args.optimizers == ["all"] else args.optimizers
    lr_scale_map = load_lr_scale_map(args.lr_scale_file)
    missing_scales = set(optimizers) - set(lr_scale_map)
    if args.lr_scale_file is not None and missing_scales:
        raise ValueError(
            "LR scale file is missing selected optimizers: "
            + ", ".join(sorted(missing_scales))
        )

    results_dir = Path(args.results_dir)
    if not args.dry_run:
        results_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = results_dir / f"manifest_{args.stage}.json"
        manifest = initialize_manifest(manifest_path, args, optimizers)
        write_manifest(manifest_path, manifest)
    else:
        manifest = {
            "protocol": "sub-one-pass",
            "stage": args.stage,
            "optimizers": optimizers,
            "invocations": [{"arguments": vars(args)}],
            "runs": [],
        }

    stop = False
    for seed in args.seeds:
        for optimizer in optimizers:
            scales = [lr_scale_map.get(optimizer, value) for value in args.lr_scales]
            for lr_scale in scales:
                command = command_for(args, optimizer, repo_root, seed, lr_scale)
                experiment_name = command[command.index("--experiment_name") + 1]
                experiment_dir = results_dir / experiment_name
                summary_path = experiment_dir / "summary.json"
                log_path = experiment_dir / "run.log"
                print(
                    f"\n[{optimizer} seed={seed} lr_scale={lr_scale:g}] "
                    f"{' '.join(command)}",
                    flush=True,
                )
                run_record = {
                    "optimizer": optimizer,
                    "seed": seed,
                    "lr_scale": lr_scale,
                    "command": command,
                    "experiment_dir": str(experiment_dir),
                }

                if args.dry_run:
                    run_record["status"] = "dry-run"
                    manifest["runs"].append(run_record)
                    continue

                run_record = upsert_run_record(manifest, run_record)
                if summary_path.is_file() and not args.rerun_completed:
                    run_record["status"] = "skipped-completed"
                    write_manifest(manifest_path, manifest)
                    continue

                run_record["status"] = "running"
                run_record["started_at_unix"] = time.time()
                write_manifest(manifest_path, manifest)
                returncode = run_and_tee(command, repo_root, log_path)
                run_record["returncode"] = returncode
                run_record["finished_at_unix"] = time.time()
                run_record["status"] = "completed" if returncode == 0 else "failed"
                write_manifest(manifest_path, manifest)
                if returncode != 0 and args.fail_fast:
                    stop = True
                    break
            if stop:
                break
        if stop:
            break

    failed = [run for run in manifest["runs"] if run["status"] == "failed"]
    if failed:
        print(
            "Failed optimizers: " + ", ".join(run["optimizer"] for run in failed),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
