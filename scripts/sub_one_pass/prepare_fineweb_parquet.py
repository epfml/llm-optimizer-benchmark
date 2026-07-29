#!/usr/bin/env python3
"""Tokenize local FineWeb parquet shards into fixed uint16 token-bin files."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tiktoken
from prepare_fineweb_stream import write_tokens


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-tokens", type=int, default=536_870_912)
    parser.add_argument("--val-tokens", type=int, default=8_388_608)
    parser.add_argument("--batch-rows", type=int, default=1_024)
    return parser.parse_args()


def documents(paths, batch_rows):
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["text"], batch_size=batch_rows):
            for text in batch.column(0).to_pylist():
                yield {"text": text}


def sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    if args.train_tokens <= 0 or args.val_tokens <= 0:
        raise ValueError("Token counts must be positive.")
    missing = [str(path) for path in args.source if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing parquet sources: " + ", ".join(missing))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_final = args.output_dir / "train.bin"
    val_final = args.output_dir / "val.bin"
    manifest_path = args.output_dir / "manifest.json"
    if train_final.exists() or val_final.exists():
        raise FileExistsError(f"{args.output_dir} already contains token files.")
    train_partial = train_final.with_suffix(".bin.partial")
    val_partial = val_final.with_suffix(".bin.partial")

    tokenizer = tiktoken.get_encoding("gpt2")
    train_memmap = np.memmap(
        train_partial, dtype=np.uint16, mode="w+", shape=(args.train_tokens,)
    )
    val_memmap = np.memmap(
        val_partial, dtype=np.uint16, mode="w+", shape=(args.val_tokens,)
    )
    document_count, boundary_discarded = write_tokens(
        documents(args.source, args.batch_rows),
        tokenizer,
        val_memmap,
        train_memmap,
    )
    train_memmap.flush()
    val_memmap.flush()
    del train_memmap, val_memmap
    os.replace(train_partial, train_final)
    os.replace(val_partial, val_final)

    sources = [
        {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in args.source
    ]
    manifest = {
        "dataset": "HuggingFaceFW/fineweb",
        "config": "sample-10BT",
        "sources": sources,
        "tokenizer": "gpt2",
        "train_tokens": args.train_tokens,
        "val_tokens": args.val_tokens,
        "documents_consumed": document_count,
        "split_unit": "source_document",
        "validation_boundary_discarded_tokens": boundary_discarded,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
