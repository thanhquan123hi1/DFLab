# DFLab

Deepfake detection experiments built on CLIP ViT-L/14 and the DeepfakeBench pipeline.

## Models

| Detector | Backbone tuning | Decision fusion |
| --- | --- | --- |
| `ln_sspanet_mil` | LayerNorm | Handcrafted BDG |
| `bias_sspanet_mil` | Bias parameters (BitFit) | Handcrafted BDG |
| `bias_gmil` | Bias parameters (BitFit) | Learned gate |

Detector configurations are in `training/config/detector/`. CAMIL variants also
remain available there. Historical BiasLN figures are not results for these models.

## Environment

`requirements.txt` is the shared dependency list for the detectors. Its existing
version pins are retained for experiment reproducibility; this cleanup does not
upgrade packages or establish compatibility with newer Python/CUDA environments.
The Linux PyTorch pins use CUDA 11.3 wheels, requiring the corresponding wheel index:

```sh
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
```

For an isolated environment, `scripts/setup_colab.py` creates `.venv-colab` and
runs environment checks. Use that environment's Python for training and testing.
`install.sh` is retained as the historical setup for additional preprocessing
packages, including dlib and OpenAI CLIP; it is not required to launch experiments
in an already configured environment. Installation on a fresh machine has not
been revalidated as part of this cleanup.

## Data and paths

Use processed RGB face frames and DeepfakeBench-compatible JSON metadata.
Datasets and model weights are not redistributed in this repository.

- Set `rgb_dir`, `dataset_json_folder`, and optional `lmdb_dir` in
  `training/config/train_config.yaml` and `training/config/test_config.yaml`.
- JSON metadata determines paths, labels and dataset splits.
- For raw-video preprocessing, configure `preprocessing/config.yaml`, place the
  required landmark model in `preprocessing/dlib_tools/`, then run
  `preprocess.py` and `rearrange.py` from the `preprocessing/` directory.
- `datasets/` and `training/pretrained/` contain placement notes.

Keep checkpoint selection on source validation (`FaceForensics++`, split `val`).
Target test sets must not be used for hyperparameter selection.
Best checkpoints are saved under
`<log_dir>/<run_name>/validation/FaceForensics++/ckpt_best.pth`.
Exact optimizer-state resume and SAM/SWA are not supported by the current trainer.

## Learned fusion: bias_gmil

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

## Multi-GPU training and logs

```sh
torchrun --nproc_per_node=2 training/train.py --detector_path training/config/detector/bias_gmil.yaml --seed 1024 --ddp
```

Standalone `training/test.py` runs in a single process; it has no `--ddp` option.
Training writes JSONL and TensorBoard logs, parameter counts and gradient diagnostics.
Set the log directory for your machine, then inspect it with:

```sh
tensorboard --logdir /path/to/logs --host 127.0.0.1 --port 6006
```

## Acknowledgements

The data and experiment pipeline follows [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench).
SSPANet is kept in `training/detectors/modules/sspanet.py`; its source-integrity
check is in `tests/test_ln_sspanet_mil.py`. The bundled Face-X-ray data-generation
library retains its own [source README](training/dataset/library/README.md).
Please follow the original dataset and library terms and cite their authors.
