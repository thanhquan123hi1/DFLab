# Training and evaluation logs

`training/config/train_config.yaml` sets `log_dir`, the parent directory of training runs.
Each new run contains:

```text
<log_dir>/<model>_<seed>_<time>/
  training.log
  run.json
  history.jsonl
  ckpt_last.pth
  train/                         # TensorBoard
  validation/<dataset>/         # TensorBoard, checkpoints, predictions
  evaluation/
    <model>_<seed>_summary.log                # Unified evaluation log across all datasets with summary table
    <model>_<seed>_summary.csv                # Consolidated tabular metrics across all datasets + AVERAGE rows
    <model>_<seed>_<dataset>.csv              # Per-dataset metric breakdown
    <model>_<seed>_<dataset>.json             # Per-dataset evaluation settings & metrics metadata
    <model>_<seed>_<dataset>_predictions.npz  # Predictions, patch probabilities, attention weights
```

`run.json` combines the configuration and trainable parameter counts.
`history.jsonl` is append-only: each record has `phase=train` or `phase=validation`.
Training records retain all losses, diagnostics, gradients and metrics; validation
records contain `metrics` and `losses`. Human-readable training messages show only
key losses, AUC/EER, alpha (when available), and learning rate. Existing logs are
not migrated or deleted.

Evaluation uses the seed stored in the checkpoint. The default output is under
the current training YAML's `log_dir`, with the original run name stored in new
checkpoints. For older checkpoints, the name is inferred from the standard
`<run>/validation/<dataset>/ckpt_best.pth` or `<run>/ckpt_last.pth` path.
If the configured `log_dir` is not found on the local filesystem (e.g. running locally
after training on Colab), evaluation automatically falls back to:
`DFLab/evaluations/eval_<model>_<seed>/`.
For a checkpoint copied outside that structure, use `--output_dir` explicitly.
`--train_config` selects a different training YAML, and `--output_dir` overrides
the output location. `log_dir` (not the detector YAML's legacy `logdir`) is the
setting used by the trainer and this evaluation layout.

A single consolidated `<model>_<seed>_summary.log` records the evaluation session
across all tested datasets and ends with an ASCII summary comparison table including
an `AVERAGE` row across datasets. In addition, `<model>_<seed>_summary.csv` consolidates
all dataset rows and includes computed `AVERAGE` rows for each branch and level.
Per-dataset CSV files preserve granular metrics (0–1 scale with full precision).
NPZ files keep per-sample predictions and patch examples. Repeating evaluation of
the same model/seed/dataset in this directory replaces its reports; use a different
`--output_dir` to retain alternate evaluation settings.

This logging change does not change branch formulas. In the current evaluator,
Ensemble combines CLS and MIL. Feature-Fusion/MIL BDG comparison is not added by
the logging layout change.
