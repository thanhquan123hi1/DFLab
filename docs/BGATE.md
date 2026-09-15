# Bias/LN SSPANet bounded-gate MIL

Two isolated experimental detectors; the original detector files, train/test entry points,
configs and shell scripts are unchanged. No full-dataset performance is claimed.

## Repository layout

BGATE uses the existing repository directories, with distinct filenames:

```text
training/
  config/detector/
    bias_sspanet_bgate_mil.yaml
    ln_sspanet_bgate_mil.yaml
  detectors/bgate_mil_detector.py
  dataset/bgate_dataset.py
  trainer/bgate_trainer.py
  metrics/bgate_evaluation.py
  bgate_common.py
  bgate_logging.py
  train_bgate.py
  test_bgate.py
docs/BGATE.md
tests/test_bgate.py
```

Each YAML inherits the baseline YAML in the same detector config directory.
Use the BGATE entry points below: they import the new detectors and select the
dedicated trainer and evaluator. The baseline entry points retain their behavior.
Only generated outputs use a separate `bgate_v1` directory; there is no separate
experiments source directory.

## Architecture

The original CLIP + SSPANet feature fusion and top-16 MIL are retained.
Only the backbone adaptation differs: Bias updates biases, LN updates LayerNorm parameters.

Given feature-fusion probability `pf` and MIL probability `pl`:

```text
u = [detach(pf), detach(pl), detach(pf-pl), abs(detach(pf-pl))]
a = Linear(16,1)(GELU(Linear(4,16)(u)))
w = 0.5 + 0.25*tanh(a)
pg = (1-w)*detach(pf) + w*detach(pl)

Lbase = CE(feature_logits, y) + 0.3*BCEWithLogits(mil_logits, y)
Lgate = BCE(pg, y) + 0.1*mean((w-0.5)^2)
```

The 97-parameter gate starts at exactly 50/50. Its first epoch is frozen.
Gate loss cannot update either branch. Separate Adam optimizers and gradient clipping
keep the gate independent of baseline optimization. Gate initialization preserves the
CPU and CUDA RNG streams used by the baseline.

Primary output: `bounded_gate`. Diagnostics: `cls_only`, `feature_fusion`, `mil`,
`fixed_ensemble` (feature fusion + MIL at 50/50). The evaluator never mixes MIL into
the already gated prediction a second time. `cls_only` is an inference ablation,
not a model trained without MIL.

## Train

Run commands from the repository root in the existing baseline Python environment.
The configs inherit the corresponding baseline YAML; resolved configs are saved per run.
Paths default to the baseline Colab paths. Override them explicitly for another machine.

Bias:

```bash
python training/train_bgate.py --config training/config/detector/bias_sspanet_bgate_mil.yaml --seed 1024 --output-root /content/drive/MyDrive/NCKH_BGATE
```

LN:

```bash
python training/train_bgate.py --config training/config/detector/ln_sspanet_bgate_mil.yaml --seed 1024 --output-root /content/drive/MyDrive/NCKH_BGATE
```

For other data locations add:

```text
--rgb-dir /path/to/rgb --dataset-json-folder /path/to/dataset_json
```

Additional options: `--batch-size`, `--eval-batch-size`, `--workers`, `--epochs`,
`--device cpu|cuda`, `--deterministic`. Keep settings identical across matched seeds.
`--deterministic` enables strict PyTorch deterministic algorithms and disables cuDNN
benchmarking; it can reject unsupported operations. Without it, baseline cuDNN
benchmark behavior is retained. Do not claim exact cross-hardware reproducibility.

This version supports one device per run, baseline frame datasets and Adam/cosine.
It intentionally rejects DDP, SWA, alternate dataset types and detector warm-starts.
CLIP pretrained initialization is retained. There is no need to install a new gate library.

Recommended matched pilot seeds: `1024`, `7749`, `8386`. Final seeds additionally:
`2026`, `3407`. Run each separately; do not ensemble models from multiple seeds when
claiming stability of individual training runs.

## Resume

```bash
python training/train_bgate.py --resume /absolute/run/checkpoints/last.pt
```

Epoch-boundary resume restores both optimizers, scheduler, RNG, best score and config.
The original run, configuration, source hashes and dataset manifests must match.
Training overrides are rejected; changes to experiment code require a new run.
Only one trainer can use a run at a time. If a process is forcibly killed, its
`.training.lock` may remain: remove that specific lock only after confirming the old
trainer has stopped. Mid-epoch exact resume is not supported. If an epoch was interrupted,
its uncheckpointed log rows remain as audit history; `resume_events.jsonl` records the
restart point. A run moved elsewhere cannot be resumed in place by this v1 command.

## Test

```bash
python training/test_bgate.py --weights /absolute/run/checkpoints/best.pt --datasets Celeb-DF-v2 DFDC
```

Override data paths as for train. `--batch-size`, `--workers`, `--device`, and
`--evaluation-seed` affect evaluation only. The actual training seed and gate architecture
come from the checkpoint, not a default YAML. Only load trusted PyTorch checkpoints.

For an exported checkpoint outside its run directory, supply `--output-root`.
Use `--max-samples 64` only for a smoke evaluation; outputs are explicitly marked partial.
Every evaluation receives a fresh directory, even when re-evaluating the same checkpoint.

## Outputs

```text
OUTPUT_ROOT/bgate_v1/MODEL/seed_SEED/TIMESTAMP_UNIQUE_ID/
  run.json                         # merged config, manifests, source hashes, environment
  trainable_parameters.json
  train/training.log
  train/steps.jsonl                 # sampled step losses and group gradient norms
  train/epochs.jsonl                # weighted loss averages, gate stats, throughput/VRAM
  train/resume_events.jsonl         # created when resuming
  train/tensorboard/
  validation/metrics.jsonl          # five branches, same checkpoint
  validation/predictions_best.npz
  checkpoints/best.pt
  checkpoints/last.pt
  tests/EVALUATION_ID/
    evaluation.json
    testing.log
    summary.csv
    DATASET/metrics.json
    DATASET/predictions.npz
```

No output is written into baseline runs. A dedicated `bgate_v1` namespace is always
appended to the supplied root. Dataset load errors propagate rather than silently dropping
failed samples. Manifests record ordered paths/labels, not hashes of all image contents;
preserve the original dataset files as well.

BGATE controls Albumentations 2's private RNG per augmented frame using the seeded Python
RNG (including inside workers). Transform definitions are unchanged, but augmentation
draws can differ from historical baseline runs that did not seed Albumentations 2.
This fix is isolated to BGATE; do not attribute every historical-run difference to gating.
Epoch logs report frame metrics for all five outputs; video metrics are computed during
validation and test. Undefined selection AUC is an error rather than a successful run
without a best checkpoint.

Selection is FF++ **validation bounded-gate video AUC**, ties retain the earlier checkpoint.
This differs from historical baseline frame-AUC selection and must be disclosed.
At evaluation, each frame output is averaged per video using the existing metric utility.
CDF/DFDC are not used for training or checkpoint selection. A best checkpoint from the
warm-up epoch correctly evaluates as a fixed 50/50 gate.

## Verification

```bash
python -m pytest tests/test_bgate.py tests/test_ln_sspanet_mil.py -q
```

Tests use a tiny differentiable backbone without downloading CLIP. They cover unchanged
baseline outputs/RNG, fine-tuning masks, warm-up, gradient isolation, independent optimizer
updates, five-branch evaluation, checkpoint/resume, CLI outputs and log isolation.
These checks do not establish full CLIP throughput, dataset AUC or 96% performance.
