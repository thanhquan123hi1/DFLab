"""Verify imports and the actual RGB/JSON augmentation path, without downloading CLIP."""
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))


def main():
    if not (3, 8) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError('DFLab environment requires Python 3.8-3.12')
    required = ['numpy', 'cv2', 'torch', 'torchvision', 'albumentations', 'yaml', 'PIL', 'transformers']
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Error: Missing required packages: {missing}. Please install dependencies via: pip install -r requirements.txt")
        sys.exit(1)
    import numpy as np
    import cv2
    import torch
    import torchvision
    import transformers
    import albumentations
    import yaml
    from PIL import Image
    from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
    from detectors import DETECTOR
    from trainer.trainer import Trainer
    for name in ('torch', 'torchvision', 'transformers', 'numpy', 'scipy',
                 'scikit-learn', 'scikit-image', 'albumentations'):
        try:
            print(f'{name}: {importlib.metadata.version(name)}', flush=True)
        except Exception:
            pass
    cfg_path = ROOT / 'training/config/detector/ln_sspanet_mil.yaml'
    if not cfg_path.exists():
        cfg_path = ROOT / 'training/config/detector/biasln.yaml'
    with cfg_path.open() as stream:
        config = yaml.safe_load(stream)
    with tempfile.TemporaryDirectory(prefix='dflab-check-') as directory:
        folder = Path(directory)
        frames = []
        for i in range(3):
            name = f'{i}.png'
            Image.fromarray(np.random.default_rng(i).integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(folder / name)
            frames.append(name)
        metadata = {'Fixture': {'real': {split: {'video': {'label': 'real', 'frames': frames}}
                                        for split in ('train', 'test')}}}
        (folder / 'Fixture.json').write_text(json.dumps(metadata))
        config.update(rgb_dir=directory, dataset_json_folder=directory, lmdb=False,
                      train_dataset=['Fixture'], test_dataset='Fixture', label_dict={'real': 0})
        for mode in ('train', 'test'):
            dataset = DeepfakeAbstractBaseDataset(config, mode=mode)
            batch = dataset.collate_fn([dataset[i] for i in range(len(dataset))])
            assert batch['image'].shape == (3, 3, 224, 224)
            assert torch.isfinite(batch['image']).all()
    print(f'RGB/JSON train and test smoke checks passed. CUDA available: {torch.cuda.is_available()}')


if __name__ == '__main__':
    main()
