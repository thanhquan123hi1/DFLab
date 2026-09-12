# Verification (2026-09-12)

## Sampling update (2026-09-13)

- Restored DeepfakeBench's frame-count capping and sampling order as requested.
- 19 tests passed, including train/evaluation JSON fixtures with frame limits below,
  equal to, and above the available frame count. No real-dataset training was run.
- Run with `sys.path.insert(0, 'tmp/testdeps')` instead of the historical append
  command below so the tested Albumentations 1.3.1 takes precedence. The current
  global Albumentations installation fails during legacy augmentation construction.

Implemented in the main `biasln` detector. No full dataset training or AUC improvement is claimed.

## Passed

- `14 passed` in the targeted pytest suite (~17 seconds).
- Verbatim SHA256 verification of the author's official SSPANet demo at the pinned commit.
- Signed feature inputs, shape preservation, finite backward, residual multiplier [1,2].
- Official block at 1024 channels: exactly 7,345,252 parameters.
- Separate CUDA smoke: input/output `[2,1024,16,16]`, finite outputs and input gradients.
- LN-only parameter selection; no backbone attention/MLP bias gradient; active SSPA, head, MIL and fusion gradients.
- Top-k arithmetic/gradient and invalid-k rejection; mixed, all-real, all-fake and multiclass-to-binary batches.
- Real Hugging Face CLIPVisionModel with a small random configuration, including early-layer LN gradient. No pretrained CLIP weights were downloaded.
- CPU trainer epoch, validation, TensorBoard/JSON logging, strict save/reload, capped standalone evaluation and patch export on synthetic data.
- Frame/video grouping with colliding basenames, single-class metrics, empty/misaligned/nonfinite rejection cases covered by the suite where applicable.
- Actual image/JSON fixture: validation split, uniform frame sampling and fail-fast on a missing image (historical run; sampling was changed back to DeepfakeBench on 2026-09-13).
- Python compile checks and `git diff --check`.

## Not yet verified

- Full pretrained CLIP-L/14 end-to-end pilot with real FF++ and target test datasets.
- Real-dataset throughput/VRAM, cross-dataset accuracy, actual forgery localization and hyperparameter optimality.
- Multi-GPU DDP behavior; only its synchronization/unwrap path has been inspected.
- A clean install of the proposed `requirements-biasln.txt` environment; it is not a lockfile of this host.
- Pixel-mask supervision, SAM/SWA, exact optimizer/RNG resume. These are not implemented in the new pipeline.

## Local test environment

Python 3.12; host PyTorch 2.6.0+cu124. Transformers 4.44.2 and supplementary dataset dependencies were installed under ignored `tmp/testdeps`, without changing the host's global Python packages. Full tests used:

Observed versions: torchvision 0.21.0+cu124, NumPy 2.5.2, scikit-learn 1.9.0, albumentations 1.3.1, scikit-image 0.26.0, TensorBoard 2.21.0, pytest 9.1.1. The proposed clean requirements deliberately use a more conservative NumPy/scikit-image range for the legacy augmentation pipeline; this clean combination has not been installed or tested here.

```powershell
python -c "import sys; sys.path.append('tmp/testdeps'); import pytest; raise SystemExit(pytest.main(['tests/test_ln_sspanet_mil.py','-q']))"
```

Without those dependencies available, two integration tests intentionally skip; install the dependencies before interpreting a default pytest run as complete. The targeted suite is not a claim that every historical dataset/analysis utility in the repository has been tested.

The workspace has no dataset frames/JSON suitable for a real pilot at the configured Colab/Kaggle paths. Set those paths and verify the source `val` split before launching training. The new preprocessing/scheduler fixes require rerunning the baseline; historical checkpoint numbers are not controlled comparisons.
