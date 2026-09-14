# bias_gmil

`BiasGMILDetector` inherits `BiasSSPANetMILDetector`. It preserves BitFit,
SSPANet and top-k MIL and replaces the handcrafted BDG rule with a 5-input,
16-hidden-unit MLP. Its final layer starts at zero, giving MIL weight 0.5.
Routing features are detached; fusion probabilities retain gradients to both
branches. Only backbone biases remain trainable, as in the parent model.

Loss: CLS CE + 0.3 MIL BCE + lambda_fusion fusion BCE. The initial
lambda_fusion is 1.0. Fusion BCE uses the same class weights as the branches.
The gate uses signed CLS/MIL probability disagreement, both branch logits,
top-k logit standard deviation, and the top-k versus remaining-patch mean gap.
When every patch is selected, that gap is defined as zero.

## Train (from the repository root)

Set dataset paths in `training/config/train_config.yaml` and
`training/config/test_config.yaml` for your machine. Retain source-only
validation. The new detector config selects checkpoints by fusion video AUC.

```sh
python training/train.py --detector_path training/config/detector/bias_gmil.yaml --seed 1024
```

After validating the setup and fixing hyperparameters on source validation,
repeat with seeds 2026, 7749 and 8386. No target-test tuning is intended.
This command trains a new model; class inheritance does not automatically
load an old experiment's checkpoint.

## Test

Replace the checkpoint placeholder with the actual saved checkpoint:

```sh
python training/test.py --detector_path training/config/detector/bias_gmil.yaml --weights_path /path/to/ckpt_best.pth --test_dataset Celeb-DF-v2 DFDC UADFV DFDCP --ensemble_weight 0.5
```

The primary score is learned Fusion. The report also shows CLS, MIL and
Ensemble AUC at frame and video level. JSON contains the other metrics for
each branch. NPZ saves frame probabilities, image paths and gate weights;
video scores are means over frames sharing the same video parent directory.
Gate standard deviation, percentiles and mean on disagreeing predictions
are saved in the evaluation JSON. Validation NPZ also saves gate weights.

Compare branches within the same checkpoint and rerun the old model with
the same video-AUC selection protocol for a controlled experiment.
Improvement is an experimental hypothesis, not a guaranteed outcome.

## Verification

```sh
python -m pytest tests/test_bias_gmil.py tests/test_ln_sspanet_mil.py tests/test_log_naming.py -q
```

Tests use a tiny backbone to verify initialization, fusion-only gradients,
learning, checkpoint roundtrip, detached routing and evaluation. They do not
establish performance on real datasets.
