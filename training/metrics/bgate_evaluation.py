"""Five explicit outputs. Never ensemble the gated output again."""
import numpy as np
import torch
from metrics.utils import get_test_metrics
from detectors.bgate_mil_detector import BRANCHES
from bgate_common import move
from bgate_logging import gate_stats


@torch.no_grad()
def evaluate(model, loader, device, max_samples=None):
    if not isinstance(loader.sampler, torch.utils.data.SequentialSampler) or loader.drop_last:
        raise ValueError('Evaluation requires sequential sampling without drop_last to align video IDs')
    if max_samples is not None and max_samples < 1:
        raise ValueError('max_samples must be positive')
    model.eval()
    values = {name+'_prob': [] for name in BRANCHES}
    values.update(label=[], gating_w=[])
    count = 0
    for batch in loader:
        size = len(batch['label'])
        if max_samples is not None:
            size = min(size, max_samples-count)
        if size <= 0:
            break
        batch = {k: v[:size] if torch.is_tensor(v) else v for k,v in batch.items()}
        data = move(batch, device)
        output = model(data, inference=True)
        values['label'].append((data['label'] != 0).long().cpu().numpy())
        for name in BRANCHES:
            values[name+'_prob'].append(output[name+'_prob'].cpu().numpy())
        values['gating_w'].append(output['gating_w'].cpu().numpy())
        count += size
    if count == 0:
        raise ValueError('Empty evaluation')
    if max_samples is None and count != len(loader.dataset):
        raise ValueError('Incomplete evaluation: sample count differs from dataset')
    arrays = {k: np.concatenate(v) for k,v in values.items()}
    names = [str(p) for p in loader.dataset.data_dict['image'][:count]]
    arrays['image_names'] = np.asarray(names)
    arrays['video_ids'] = np.asarray([p.replace('\\','/').rsplit('/',1)[0] for p in names])
    metrics = {}
    for name in BRANCHES:
        result = get_test_metrics(arrays[name+'_prob'], arrays['label'], names)
        metrics[name] = {k:v for k,v in result.items() if k not in ('pred','label')}
    stats = gate_stats(arrays['gating_w'])
    for label, name in ((0,'real'), (1,'fake')):
        subset = arrays['gating_w'][arrays['label'] == label]
        if len(subset):
            stats[name+'_mean'] = float(subset.mean())
    return dict(branches=metrics, gate=stats, partial=max_samples is not None,
                evaluated_frames=count, available_frames=len(loader.dataset)), arrays
