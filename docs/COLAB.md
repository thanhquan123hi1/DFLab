# DFLab on Colab

Select a GPU runtime with Python 3.10, 3.11 or 3.12. Colab's preinstalled packages
change over time; DFLab installs its own environment instead of downgrading the
notebook's NumPy, OpenCV or CUDA-enabled PyTorch. Do not use the old Python 3.8
installation instructions. Setup downloads a separate PyTorch stack and can take
several minutes. Recreate this environment when Colab allocates a new runtime.

## 1. Clone and install

```python
!git clone https://github.com/thanhquan123hi1/DFLab.git /content/DFLab
%cd /content/DFLab
import sys, subprocess
print(sys.executable, sys.version)
subprocess.run([sys.executable, "scripts/setup_colab.py"], check=True)
```

Setup stops immediately on unsupported Python. If Python is 3.8, use a fresh
supported runtime; installing NumPy 1.26 into Python 3.8 cannot repair it.
The environment does not expose system site packages. Python's `-I` option also
ignores inherited `PYTHONPATH`/`PYTHONHOME` and user-site packages when running.
No kernel restart is needed because training runs in a separate Python process.

## 2. Set dataset paths

Place processed RGB frames and JSON metadata on local runtime storage for better
I/O. Mount Drive if storing checkpoints there:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Edit `training/config/train_config.yaml` and `training/config/test_config.yaml`:

- `rgb_dir`: root used for relative image paths in JSON.
- `dataset_json_folder`: directory containing files such as `FaceForensics++.json`.
- `log_dir` in the training config: a persistent output directory on mounted Drive.

The loader reads paths from JSON; it does not discover images by scanning folders.
Ensure those paths resolve under `rgb_dir`, or use valid absolute image paths.
Training selects the first 8 numerically sorted frames per video; validation/test
select the first 32. FF++ source validation uses the JSON `val` split.

## 3. Check, then train

```python
PYTHON = '/content/DFLab/.venv-colab/bin/python'
subprocess.run([PYTHON, '-I', 'scripts/check_environment.py'], check=True)
subprocess.run([PYTHON, '-I', '-m', 'pytest', 'tests', '-q'], check=True)
subprocess.run([
    PYTHON, '-I', 'training/train.py',
    '--detector_path', 'training/config/detector/biasln.yaml',
    '--train_dataset', 'FaceForensics++',
], check=True)
```

The environment check reads generated PNG/JSON samples through the real train/test
loader and augmentation pipeline. It does not validate your real dataset or download
pretrained CLIP. The first training run downloads CLIP weights from Hugging Face.
If the default batch size exceeds GPU memory, lower `train_batchSize` and
`test_batchSize` in the detector YAML; this is an experiment setting, not an
automatic change made by the installer.

## 4. Test a selected checkpoint

```python
subprocess.run([
    PYTHON, '-I', 'training/test.py',
    '--detector_path', 'training/config/detector/biasln.yaml',
    '--weights_path', '/path/to/run/validation/FaceForensics++/ckpt_best.pth',
    '--test_dataset', 'Celeb-DF-v2',
], check=True)
```

Use the environment's Python for every command, not `!python` or `/usr/bin/python3`.

## Compatibility scope

The installer pins NumPy 1.26.4, OpenCV headless 4.10.0.84, SciPy 1.14.1,
scikit-learn 1.5.2, scikit-image 0.24.0, Albumentations 1.3.1, Transformers 4.44.2,
PyTorch 2.6.0 and torchvision 0.21.0. Keeping Albumentations 1.x avoids changing
augmentation APIs, interpolation defaults or random behavior as part of setup.
The architecture, pretrained CLIP identifier, LN tuning, SSPANet source bytes,
CE/MIL objectives, normalization and sampling are unchanged by this environment update.

Legacy preprocessing/alternative dataset utilities are outside this dependency
recipe. The supported workflow is the current `biasln` detector with RGB frames
and JSON metadata. Local checks do not establish real-data accuracy, GPU capacity
or successful execution on a hosted Colab session.
