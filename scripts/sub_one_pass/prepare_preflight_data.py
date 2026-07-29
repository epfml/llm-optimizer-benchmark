#!/usr/bin/env python3
"""Create a small deterministic token-bin for optimizer compatibility checks."""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-tokens", type=int, default=4_194_304)
    parser.add_argument("--val-tokens", type=int, default=524_288)
    parser.add_argument("--vocab-size", type=int, default=50_304)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def write_random_tokens(path, count, vocab_size, rng, chunk_size=1_048_576):
    token_file = np.memmap(path, dtype=np.uint16, mode="w+", shape=(count,))
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        token_file[start:end] = rng.integers(
            0, vocab_size, size=end - start, dtype=np.uint16
        )
    token_file.flush()
    del token_file


def main():
    args = parse_args()
    if args.train_tokens <= 0 or args.val_tokens <= 0:
        raise ValueError("Token counts must be positive.")
    if not 1 < args.vocab_size <= np.iinfo(np.uint16).max:
        raise ValueError("vocab-size must fit uint16 and be greater than one.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"
    manifest_path = output_dir / "manifest.json"
    if train_path.exists() or val_path.exists():
        raise FileExistsError(f"{output_dir} already contains token files.")

    rng = np.random.default_rng(args.seed)
    write_random_tokens(train_path, args.train_tokens, args.vocab_size, rng)
    write_random_tokens(val_path, args.val_tokens, args.vocab_size, rng)
    manifest = {
        "purpose": "optimizer-compatibility-preflight-only",
        "synthetic": True,
        "seed": args.seed,
        "vocab_size": args.vocab_size,
        "train_tokens": args.train_tokens,
        "val_tokens": args.val_tokens,
        "eligible_for_scientific_comparison": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
