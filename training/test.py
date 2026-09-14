"""Strict checkpoint evaluation, frame/video metrics, and patch diagnostics."""
import argparse
import os
import pickle
import random
import sys
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'training'))
from detectors import DETECTOR
from metrics.utils import binary_metrics, get_test_metrics, write_json, format_compact_test_report


@torch.no_grad()
def evaluate(model, loader, device, max_samples=None, patch_limit=32, save_feat=False, ensemble_weight=0.5):
    values = {k: [] for k in ('prob', 'label', 'label_spe', 'cls_only_prob', 'mil_prob', 'feat')}
    patches, names, patch_names, attns = [], [], [], []
    count = 0
    global_names = loader.dataset.data_dict['image']
    for batch in loader:
        remaining = len(batch['label']) if max_samples is None else min(len(batch['label']), max_samples - count)
        if remaining <= 0:
            break
        data = {k: v[:remaining].to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(data, inference=True)
        batch_names = global_names[count:count+remaining]
        names.extend(batch_names)
        values['label'].append((data['label'] != 0).long().cpu().numpy())
        values['label_spe'].append(data.get('label_spe', data['label']).cpu().numpy())
        for key in ('prob', 'cls_only_prob', 'mil_prob', 'feat'):
            if key in out and (key != 'feat' or save_feat):
                values[key].append(out[key].cpu().numpy())
        take = min(remaining, max(0, patch_limit - len(patch_names)))
        if take and 'patch_logits' in out:
            patches.append(out['patch_logits'][:take].sigmoid().cpu().numpy())
            patch_names.extend(batch_names[:take])
        if take and 'attn_weights' in out:
            attns.append(out['attn_weights'][:take].cpu().numpy())
        count += remaining
    arrays = {k: np.concatenate(v) for k, v in values.items() if v}
    if not count:
        raise ValueError('No samples evaluated')
    result = get_test_metrics(arrays['prob'], arrays['label'], names)
    for key in ('cls_only_prob', 'mil_prob'):
        if key in arrays:
            branch = get_test_metrics(arrays[key], arrays['label'], names)
            result.update({key.replace('_prob', '') + '_' + k: v for k, v in branch.items() if k not in ('pred', 'label')})
    if 'mil_prob' in arrays and 'prob' in arrays and ensemble_weight is not None and ensemble_weight >= 0:
        ens_w = float(ensemble_weight)
        ens_prob = (1.0 - ens_w) * arrays['prob'] + ens_w * arrays['mil_prob']
        arrays['ensemble_prob'] = ens_prob
        ens_branch = get_test_metrics(ens_prob, arrays['label'], names)
        result.update({'ensemble_' + k: v for k, v in ens_branch.items() if k not in ('pred', 'label')})
    arrays.update(image_names=np.asarray(names), patch_image_names=np.asarray(patch_names),
                  patch_prob=np.concatenate(patches) if patches else np.empty((0,)),
                  attn_weights=np.concatenate(attns) if attns else np.empty((0,)))
    return result, arrays


def safe_torch_load(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detector_path', default='training/config/detector/ln_sspanet_mil.yaml')
    parser.add_argument('--weights_path', required=True)
    parser.add_argument('--test_dataset', nargs='+')
    parser.add_argument('--save_feat', action='store_true')
    parser.add_argument('--feat_out_dir', default='tsne_pkls')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--max_samples', type=int)
    parser.add_argument('--patch_limit', type=int, default=32)
    parser.add_argument('--ensemble_weight', type=float, default=0.5,
                        help='Weight for MIL branch in decision ensemble: (1-w)*Fusion + w*MIL (default: 0.5, negative to disable)')
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error('--max_samples must be positive')
    with open(args.detector_path, encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    with open(os.path.join(ROOT, 'training/config/test_config.yaml'), encoding='utf-8') as stream:
        config.update(yaml.safe_load(stream))
    checkpoint = safe_torch_load(args.weights_path, map_location='cpu')
    # Preserve model architecture from the checkpoint, but use current evaluation paths.
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        for key in ('clip_model_name', 'use_patch', 'use_sspanet', 'lambda_mil', 'mil_topk',
                    'fusion_alpha_init', 'fusion_gamma_init', 'cross_attn_heads', 'cross_attn_dropout',
                    'label_smoothing', 'weight_real', 'weight_fake',
                    'resolution', 'mean', 'std', 'model_name'):
            if key in checkpoint['config']:
                config[key] = checkpoint['config'][key]
    config['eval_split'] = 'test'
    datasets = args.test_dataset or config['test_dataset']
    seed = config.get('manualSeed', 1024)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if config.get('cuda', True) and torch.cuda.is_available() else 'cpu')
    model = DETECTOR[config['model_name']](config).to(device)
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    # Allow loading checkpoints from older versions that included fusion_alpha
    model_keys = set(model.state_dict().keys())
    filtered_state = {k: v for k, v in state.items() if k in model_keys}
    model.load_state_dict(filtered_state, strict=True)
    model.eval()
    from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
    out_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(args.weights_path)), 'evaluation')
    os.makedirs(out_dir, exist_ok=True)
    write_json(os.path.join(out_dir, 'evaluation_config.json'), dict(config=config, checkpoint=os.path.abspath(args.weights_path), max_samples=args.max_samples))
    ens_w = args.ensemble_weight if args.ensemble_weight >= 0 else None
    for name in datasets:
        cfg = dict(config, test_dataset=name)
        dataset = DeepfakeAbstractBaseDataset(cfg, mode='test')
        loader = DataLoader(dataset, batch_size=config['test_batchSize'], shuffle=False,
                            num_workers=config['workers'], collate_fn=dataset.collate_fn, drop_last=False)
        result, arrays = evaluate(model, loader, device, args.max_samples, args.patch_limit, args.save_feat, ensemble_weight=ens_w)
        safe = name.replace('/', '_').replace('\\', '_')
        write_json(os.path.join(out_dir, safe + '_metrics.json'), result)
        np.savez_compressed(os.path.join(out_dir, safe + '_predictions.npz'), **arrays)
        print(format_compact_test_report(name, result))
        if args.save_feat:
            os.makedirs(args.feat_out_dir, exist_ok=True)
            payload = dict(feat=arrays['feat'], label=arrays['label'], label_spe=arrays['label_spe'],
                           pred=arrays['prob'], img_names=arrays['image_names'].tolist(),
                           dataset=name, weights_path=args.weights_path, model_name=config['model_name'])
            with open(os.path.join(args.feat_out_dir, f'tsne_dict_{config["model_name"]}_{safe}.pkl'), 'wb') as stream:
                pickle.dump(payload, stream)


if __name__ == '__main__':
    main()
