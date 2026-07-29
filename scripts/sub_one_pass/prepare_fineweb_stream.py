#!/usr/bin/env python3
"""Stream a fixed-size FineWeb corpus directly into uint16 token files."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="sample-10BT")
    parser.add_argument("--train-tokens", type=int, default=1_073_741_824)
    parser.add_argument("--val-tokens", type=int, default=8_388_608)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2357)
    return parser.parse_args()


def write_tokens(stream, tokenizer, val_memmap, train_memmap):
    val_written = 0
    train_written = 0
    validation_boundary_discarded_tokens = 0
    for document_count, example in enumerate(stream, start=1):
        token_ids = tokenizer.encode_ordinary(example["text"])
        token_ids.append(tokenizer.eot_token)
        tokens = np.asarray(token_ids, dtype=np.uint16)

        if val_written < len(val_memmap):
            count = min(len(tokens), len(val_memmap) - val_written)
            val_memmap[val_written : val_written + count] = tokens[:count]
            val_written += count
            if val_written == len(val_memmap):
                # Never split one source document across validation and train.
                validation_boundary_discarded_tokens = len(tokens) - count
            tokens = np.empty(0, dtype=np.uint16)
        elif train_written < len(train_memmap):
            count = min(len(tokens), len(train_memmap) - train_written)
            train_memmap[train_written : train_written + count] = tokens[:count]
            train_written += count

        if document_count % 1_000 == 0:
            print(
                f"documents={document_count:,} "
                f"val_tokens={val_written:,}/{len(val_memmap):,} "
                f"train_tokens={train_written:,}/{len(train_memmap):,}",
                flush=True,
            )
            val_memmap.flush()
            train_memmap.flush()

        if val_written == len(val_memmap) and train_written == len(train_memmap):
            return document_count, validation_boundary_discarded_tokens

    raise RuntimeError(
        "FineWeb stream ended before the requested token budgets were filled."
    )


def main():
    import tiktoken
    from datasets import load_dataset

    args = parse_args()
    if args.train_tokens <= 0 or args.val_tokens <= 0:
        raise ValueError("Token counts must be positive.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_final = output_dir / "train.bin"
    val_final = output_dir / "val.bin"
    manifest_path = output_dir / "manifest.json"
    if train_final.exists() or val_final.exists():
        raise FileExistsError(
            f"{output_dir} already contains train.bin or val.bin; "
            "choose a new output directory."
        )

    train_partial = output_dir / "train.bin.partial"
    val_partial = output_dir / "val.bin.partial"
    tokenizer = tiktoken.get_encoding("gpt2")
    stream = load_dataset(
        "HuggingFaceFW/fineweb",
        name=args.config,
        split="train",
        streaming=True,
    )
    stream = stream.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    train_memmap = np.memmap(
        train_partial, dtype=np.uint16, mode="w+", shape=(args.train_tokens,)
    )
    val_memmap = np.memmap(
        val_partial, dtype=np.uint16, mode="w+", shape=(args.val_tokens,)
    )
    (
        document_count,
        validation_boundary_discarded_tokens,
    ) = write_tokens(stream, tokenizer, val_memmap, train_memmap)
    train_memmap.flush()
    val_memmap.flush()
    del train_memmap, val_memmap
    os.replace(train_partial, train_final)
    os.replace(val_partial, val_final)

    manifest = {
        "dataset": "HuggingFaceFW/fineweb",
        "config": args.config,
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "tokenizer": "gpt2",
        "train_tokens": args.train_tokens,
        "val_tokens": args.val_tokens,
        "documents_consumed": document_count,
        "split_unit": "source_document",
        "validation_boundary_discarded_tokens": (validation_boundary_discarded_tokens),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
