"""Unique experiment outputs, metadata and atomic checkpoints."""
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import torch
from metrics.utils import write_json


def unique_dir(parent):
    stamp = datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S')
    path = Path(parent) / (stamp + '_' + uuid.uuid4().hex[:12])
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve()


def new_run(root, model, seed):
    # Always append a dedicated namespace, even if a baseline log root was supplied.
    return unique_dir(Path(root) / 'bgate_v1' / model / f'seed_{seed}')


def logger_for(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(str(path.resolve()))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        for handler in (logging.FileHandler(path, encoding='utf-8'), logging.StreamHandler()):
            handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
            logger.addHandler(handler)
    return logger


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(root):
    files = ([root/p for p in ('bgate_common.py', 'bgate_logging.py', 'train_bgate.py',
                       'test_bgate.py', 'detectors/bgate_mil_detector.py',
                       'trainer/bgate_trainer.py', 'dataset/bgate_dataset.py',
                       'metrics/bgate_evaluation.py')] +
             list((root/'detectors').glob('*sspanet*detector.py')) +
             [root/'detectors/modules/sspanet.py', root/'dataset/abstract_dataset.py',
              root/'dataset/albu.py', root/'metrics/utils.py'])
    return {p.relative_to(root).as_posix(): sha256(p) for p in files}


def verify_resume(original, config, manifests, sources):
    if original['config'] != config:
        raise ValueError('Run metadata and checkpoint configurations differ')
    if original['manifests'] != manifests:
        raise ValueError('Dataset manifests changed since the original run')
    if original.get('source_hashes') != sources:
        raise ValueError('Training source changed; resume would mix different implementations')


def metadata(root):
    def git(*args):
        try:
            return subprocess.check_output(['git', *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    from importlib.metadata import version, PackageNotFoundError
    versions = {}
    for name in ('torch', 'transformers', 'numpy', 'albumentations', 'scikit-learn'):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return dict(git_commit=git('rev-parse', 'HEAD'), git_status=git('status', '--porcelain'),
                git_diff=git('diff', 'HEAD'), python=platform.python_version(), versions=versions,
                cuda=torch.version.cuda)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path):
    # Only load trusted checkpoints: optimizer/RNG payloads require pickle.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    from detectors.bgate_mil_detector import ARCHITECTURE
    if checkpoint.get('architecture') != ARCHITECTURE:
        raise ValueError('Expected a BGATE v1 checkpoint; baseline resume is not supported')
    return checkpoint


def gate_stats(weights):
    w = np.asarray(weights, dtype=float)
    return dict(mean=float(w.mean()), std=float(w.std()), min=float(w.min()), max=float(w.max()),
                p5=float(np.quantile(w,.05)), p50=float(np.quantile(w,.5)), p95=float(np.quantile(w,.95)),
                near_lower=float((w<.26).mean()), near_upper=float((w>.74).mean()))
