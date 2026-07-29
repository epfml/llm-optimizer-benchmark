# Sub-one-pass optimizer benchmark

This is a new experiment and is unrelated to Datablations. It compares all 20
optimizers exposed by this repository on the same nested prefix of a large,
fixed tokenized corpus.

The estimand is early **sample efficiency**: validation quality after the same
number of observed training tokens. It is not an equal-FLOP or equal-wall-clock
comparison, so throughput, elapsed time and peak memory are reported separately.

## Fixed protocol

- Model: approximately 30M-parameter Llama (`8 × 384`, 6 heads).
- Sequence length: 512.
- Micro batch: 16; gradient accumulation: 8.
- Tokens per optimizer step: 65,536.
- Main checkpoints: 16M, 32M, 64M, 128M and 256M tokens.
- Data order: deterministic, fixed-boundary, no-replacement affine permutation.
- Schedule: 16-step linear warmup followed by constant LR.
- Optimizer-internal horizon: fixed at 256M tokens in every stage.
- Validation: the same 64 batches at every checkpoint.

With the default 1,073,741,824-token training file, the main trajectory exposes
about 25% of the unique corpus and never begins a second pass.

## Optimizers

`adamw`, `sgd`, `muon`, `soap`, `ademamix`, `lion`, `sf-adamw`, `sf-sgd`,
`signsgd`, `signum`, `prodigy`, `sophiag`, `adopt`, `mars`, `adafactor`,
`lamb`, `scion`, `scion-light`, `d-muon`, `muon-pytorch`.

`muon-pytorch` uses `torch.optim.Muon`, so all formal runs need one common
PyTorch release that provides this class. Do not run only this optimizer under a
different PyTorch version and mix it into the same ranking.

## Prepare the fixed corpus

```bash
python scripts/sub_one_pass/prepare_fineweb_stream.py \
  --output-dir /path/to/fineweb-1b
```

The command streams a bounded FineWeb sample and writes uint16 `train.bin`,
`val.bin` and `manifest.json`. It does not materialize the full upstream corpus.

The commands below abbreviate the repeated data arguments as:

```bash
DATA_ARGS="--dataset token-bin \
  --train-data-path /path/to/fineweb-1b/train.bin \
  --val-data-path /path/to/fineweb-1b/val.bin"
```

## S0: compatibility smoke

This runs 32 steps per optimizer. Run it before any expensive queue:

```bash
python scripts/sub_one_pass/run_all.py \
  --stage smoke \
  $DATA_ARGS \
  --results-dir ./exps/sub_one_pass
```

## S1: equal-budget LR screening

Each optimizer is run at `0.3×`, `1×` and `3×` its registry LR for 16M tokens:

```bash
python scripts/sub_one_pass/run_all.py \
  --stage tune \
  $DATA_ARGS \
  --results-dir ./exps/sub_one_pass
```

Select the best scale:

```bash
python scripts/sub_one_pass/select_lr.py \
  ./exps/sub_one_pass/manifest_tune.json
```

Exit status 2 means at least one optimum lies on the grid boundary. Expand that
optimizer's LR grid before starting the main experiment.

Short stages keep AdEMAMix's internal warmup fixed to the 256M protocol horizon;
they are true prefixes of the main optimizer dynamics rather than compressed
short-horizon variants.

## S2: all-optimizer main curves

```bash
python scripts/sub_one_pass/run_all.py \
  --stage main \
  --lr-scale-file ./exps/sub_one_pass/selected_lr_scales.json \
  $DATA_ARGS \
  --results-dir ./exps/sub_one_pass
```

Every run is sequential. `manifest_main.json` is atomically updated after each
state change. Restarting the same command skips directories that already contain
`summary.json`; pass `--rerun-completed` only when an intentional rerun is
required.

## S3: paired-seed confirmation

After S2, choose the prespecified leading group plus AdamW and run two additional
seeds. For example:

```bash
python scripts/sub_one_pass/run_all.py \
  --stage confirm \
  --optimizers adamw muon soap lion \
  --seeds 1 2 \
  --lr-scale-file ./exps/sub_one_pass/selected_lr_scales.json \
  $DATA_ARGS \
  --results-dir ./exps/sub_one_pass
```

## Summaries and time estimate

```bash
python scripts/sub_one_pass/summarize.py ./exps/sub_one_pass

python scripts/sub_one_pass/estimate_runtime.py \
  ./exps/sub_one_pass \
  --target-tokens 268435456
```

The runtime estimate must be generated from GPU smoke/calibration results.
CPU estimates and generic A800 estimates are not substitutes for measurement.

## Dry runs

```bash
python scripts/sub_one_pass/run_all.py --stage smoke --dry-run
python scripts/sub_one_pass/run_all.py --stage tune --dry-run
python scripts/sub_one_pass/run_all.py --stage main --dry-run
```
