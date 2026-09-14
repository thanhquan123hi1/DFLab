"""Strict checkpoint evaluation, frame/video metrics, and patch diagnostics."""
import argparse
import datetime
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
from metrics.reporting import (
    evaluation_directory,
    safe_name,
    write_metrics_csv,
    write_summary_csv,
    format_summary_table,
    FEATURE_MODELS,
)
from detectors.bias_sspanet_feat_mil_detector import compute_dynamic_gating


@torch.no_grad()
def evaluate(model, loader, device, max_samples=None, patch_limit=32, save_feat=False, ensemble_weight=0.5):
    values = {k: [] for k in ('prob', 'label', 'label_spe', 'cls_only_prob', 'mil_prob', 'feat',
                              'gating_w', 'gating_w_cls', 'gating_w_f', 'bdg_cls_mil_prob', 'bdg_f_mil_prob')}
    patches, names, patch_names, attns = [], [], [], []
    count = 0
    global_names = loader.dataset.data_dict['image']
    is_feat_model = (getattr(model, 'config', {}).get('model_name') in FEATURE_MODELS or
                     hasattr(model, 'fusion_alpha') or hasattr(model, 'fusion_gamma'))

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

        # Compute dynamic gating if not already output by model and not a learned gate model
        is_learned_gate = (getattr(model, 'config', {}).get('model_name') == 'bias_gmil' or
                           getattr(model, 'learned_gate', False))
        if not is_learned_gate:
            if 'patch_logits' in out and 'mil_prob' in out:
                gw_min = getattr(model, 'gating_w_min', None)
                if gw_min is None:
                    gw_min = getattr(model, 'config', {}).get('gating_w_min', 0.05)
                gw_max = getattr(model, 'gating_w_max', None)
                if gw_max is None:
                    gw_max = getattr(model, 'config', {}).get('gating_w_max', 0.85)
                g_tau = getattr(model, 'gating_tau', None)
                if g_tau is None:
                    g_tau = getattr(model, 'config', {}).get('gating_tau', 0.15)
                if 'bdg_cls_mil_prob' not in out and 'cls_only_prob' in out:
                    w_cls, _, _ = compute_dynamic_gating(out['cls_only_prob'], out['patch_logits'], gw_min, gw_max, g_tau)
                    out['gating_w_cls'] = w_cls
                    out['bdg_cls_mil_prob'] = (1.0 - w_cls) * out['cls_only_prob'] + w_cls * out['mil_prob']
                    if 'gating_w' not in out:
                        out['gating_w'] = w_cls
                if 'bdg_f_mil_prob' not in out and 'prob' in out and is_feat_model:
                    w_f, _, _ = compute_dynamic_gating(out['prob'], out['patch_logits'], gw_min, gw_max, g_tau)
                    out['gating_w_f'] = w_f
                    out['bdg_f_mil_prob'] = (1.0 - w_f) * out['prob'] + w_f * out['mil_prob']
            elif 'gating_w' in out and 'prob' in out and 'bdg_cls_mil_prob' not in out:
                out['bdg_cls_mil_prob'] = out['prob']
                if 'gating_w_cls' not in out:
                    out['gating_w_cls'] = out['gating_w']

        for key in ('prob', 'cls_only_prob', 'mil_prob', 'feat', 'gating_w',
                    'gating_w_cls', 'gating_w_f', 'bdg_cls_mil_prob', 'bdg_f_mil_prob'):
            if key in out and out[key] is not None and (key != 'feat' or save_feat):
                val = out[key]
                if torch.is_tensor(val):
                    val = val.cpu().numpy()
                values[key].append(val)
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

    # Construct ensemble arrays
    ens_w = float(ensemble_weight) if ensemble_weight is not None and ensemble_weight >= 0 else 0.5
    ens_w = min(max(ens_w, 0.0), 1.0)
    if 'cls_only_prob' in arrays and 'mil_prob' in arrays:
        arrays['ens_cls_mil_prob'] = (1.0 - ens_w) * arrays['cls_only_prob'] + ens_w * arrays['mil_prob']
        arrays['ensemble_prob'] = arrays['ens_cls_mil_prob']
    elif 'prob' in arrays and 'mil_prob' in arrays and 'ensemble_prob' not in arrays:
        arrays['ensemble_prob'] = (1.0 - ens_w) * arrays['prob'] + ens_w * arrays['mil_prob']

    if 'prob' in arrays and 'mil_prob' in arrays and is_feat_model:
        arrays['ens_f_mil_prob'] = (1.0 - ens_w) * arrays['prob'] + ens_w * arrays['mil_prob']

    if is_feat_model and 'prob' in arrays:
        arrays['feature_fusion_prob'] = arrays['prob']

    result = get_test_metrics(arrays['prob'], arrays['label'], names)
    if getattr(model, 'config', {}).get('model_name') == 'bias_gmil' or getattr(model, 'learned_gate', False):
        result['learned_gate'] = True
    result['is_feat_model'] = is_feat_model

    # Compute gating diagnostics
    for gw_key, prefix in [('gating_w', 'gating_w_'), ('gating_w_cls', 'gating_w_cls_'), ('gating_w_f', 'gating_w_f_')]:
        if gw_key in arrays and len(arrays[gw_key]) > 0:
            gw = arrays[gw_key]
            lbl = arrays['label']
            result[f'{prefix}mean'] = float(np.mean(gw))
            result[f'{prefix}std'] = float(np.std(gw))
            for percentile in (5, 50, 95):
                result[f'{prefix}p{percentile}'] = float(np.percentile(gw, percentile))
            if 'cls_only_prob' in arrays and 'mil_prob' in arrays:
                disagreement = (arrays['cls_only_prob'] >= .5) != (arrays['mil_prob'] >= .5)
                if disagreement.any():
                    result[f'{prefix}disagreement'] = float(gw[disagreement].mean())
            if (lbl == 0).any():
                result[f'{prefix}real'] = float(np.mean(gw[lbl == 0]))
            if (lbl == 1).any():
                result[f'{prefix}fake'] = float(np.mean(gw[lbl == 1]))
    if 'gating_w_mean' in result:
        result['gating_weight_mean'] = result['gating_w_mean']

    # Evaluate 7 ablation variants
    branch_specs = [
        ('bdg_cls_mil', 'bdg_cls_mil_prob'),
        ('bdg_f_mil', 'bdg_f_mil_prob'),
        ('ens_cls_mil', 'ens_cls_mil_prob'),
        ('ens_f_mil', 'ens_f_mil_prob'),
        ('feature_fusion', 'feature_fusion_prob' if 'feature_fusion_prob' in arrays else None),
        ('mil', 'mil_prob'),
        ('cls_only', 'cls_only_prob'),
    ]
    branch_metrics_cache = {}
    for branch_name, arr_key in branch_specs:
        if arr_key and arr_key in arrays:
            b_metrics = get_test_metrics(arrays[arr_key], arrays['label'], names)
            branch_metrics_cache[arr_key] = b_metrics
            result.update({f'{branch_name}_{k}': v for k, v in b_metrics.items() if k not in ('pred', 'label')})

    # Legacy compatibility keys (reusing cached branch metrics)
    for key in ('cls_only_prob', 'mil_prob'):
        if key in arrays:
            b_metrics = branch_metrics_cache.get(key)
            if b_metrics is None:
                b_metrics = get_test_metrics(arrays[key], arrays['label'], names)
            result.update({key.replace('_prob', '') + '_' + k: v for k, v in b_metrics.items() if k not in ('pred', 'label')})
    if 'ensemble_prob' in arrays:
        ens_metrics = branch_metrics_cache.get('ens_cls_mil_prob')
        if ens_metrics is None or arrays['ensemble_prob'] is not arrays.get('ens_cls_mil_prob'):
            ens_metrics = get_test_metrics(arrays['ensemble_prob'], arrays['label'], names)
        result.update({'ensemble_' + k: v for k, v in ens_metrics.items() if k not in ('pred', 'label')})

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
    parser.add_argument('--train_config', default=os.path.join(ROOT, 'training/config/train_config.yaml'),
                        help='Training YAML supplying the default log_dir for evaluation reports')
    parser.add_argument('--max_samples', type=int)
    parser.add_argument('--patch_limit', type=int, default=32)
    parser.add_argument('--ensemble_weight', type=float, default=0.5,
                        help='Weight for MIL branch in decision ensemble: (1-w)*CLS + w*MIL (default: 0.5, negative to disable)')
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
                    'gate_hidden_dim', 'lambda_fusion', 'gating_w_min', 'gating_w_max', 'gating_tau',
                    'resolution', 'mean', 'std', 'model_name', 'manualSeed'):
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
    state = checkpoint.get('state_dict', checkpoint)
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    # Allow loading checkpoints from older versions that included fusion_alpha
    model_keys = set(model.state_dict().keys())
    filtered_state = {k: v for k, v in state.items() if k in model_keys}
    model.load_state_dict(filtered_state, strict=True)
    model.eval()
    from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
    with open(args.train_config, encoding='utf-8') as stream:
        train_config = yaml.safe_load(stream) or {}

    configured_log_dir = train_config.get('log_dir')
    fallback_dir = os.path.join(ROOT, 'evaluations', f'eval_{safe_name(config["model_name"])}_{seed}')
    out_dir = evaluation_directory(args.weights_path, checkpoint,
                                   configured_log_dir, args.output_dir,
                                   fallback_dir=fallback_dir)
    os.makedirs(out_dir, exist_ok=True)
    if str(out_dir) == str(fallback_dir) and not (configured_log_dir and os.path.exists(str(configured_log_dir))):
        print(f"[Notice] Configured training log_dir not found on system. Saving evaluation to: {out_dir}")

    ens_w = args.ensemble_weight if args.ensemble_weight >= 0 else None
    log_stem = f'{safe_name(config["model_name"])}_{seed}'
    summary_log_path = os.path.join(out_dir, f'{log_stem}_summary.log')

    time_now_str = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    header_str = (
        f"{'='*80}\n"
        f"EVALUATION SESSION: {time_now_str}\n"
        f"Model: {config['model_name']} | Seed: {seed}\n"
        f"Checkpoint: {os.path.abspath(args.weights_path)}\n"
        f"Output directory: {os.path.abspath(out_dir)}\n"
        f"Datasets: {', '.join(datasets)}\n"
        f"Ensemble weight: {ens_w}\n"
        f"{'='*80}\n\n"
    )
    with open(summary_log_path, 'w', encoding='utf-8') as stream:
        stream.write(header_str)
    print(header_str.rstrip() + "\n")

    dataset_results = []
    for idx, name in enumerate(datasets, 1):
        cfg = dict(config, test_dataset=name)
        dataset = DeepfakeAbstractBaseDataset(cfg, mode='test')
        loader = DataLoader(dataset, batch_size=config['test_batchSize'], shuffle=False,
                            num_workers=config['workers'], collate_fn=dataset.collate_fn, drop_last=False)
        result, arrays = evaluate(model, loader, device, args.max_samples, args.patch_limit, args.save_feat, ensemble_weight=ens_w)
        dataset_results.append((name, result))
        safe = safe_name(name)
        stem = f'{safe_name(config["model_name"])}_{seed}_{safe}'
        metadata = dict(config=cfg, checkpoint=os.path.abspath(args.weights_path),
                        seed=seed, epoch=checkpoint.get('epoch'), max_samples=args.max_samples,
                        ensemble_weight=ens_w, ensemble_base='cls_only_prob',
                        evaluated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        metrics=result)
        write_json(os.path.join(out_dir, stem + '.json'), metadata)
        write_metrics_csv(os.path.join(out_dir, stem + '.csv'), result,
                          config['model_name'], seed, name, os.path.abspath(args.weights_path), ens_w)
        np.savez_compressed(os.path.join(out_dir, stem + '_predictions.npz'), **arrays)
        report = format_compact_test_report(name, result)
        print(f"[{idx}/{len(datasets)}] {report}")
        with open(summary_log_path, 'a', encoding='utf-8') as stream:
            stream.write(f"[{idx}/{len(datasets)}] Dataset: {name}\n")
            stream.write(report + '\n\n')

        if args.save_feat:
            os.makedirs(args.feat_out_dir, exist_ok=True)
            payload = dict(feat=arrays['feat'], label=arrays['label'], label_spe=arrays['label_spe'],
                           pred=arrays['prob'], img_names=arrays['image_names'].tolist(),
                           dataset=name, weights_path=args.weights_path, model_name=config['model_name'])
            with open(os.path.join(args.feat_out_dir, f'tsne_dict_{config["model_name"]}_{safe}.pkl'), 'wb') as stream:
                pickle.dump(payload, stream)

    # Format and save consolidated summary table and summary CSV
    summary_table = format_summary_table(dataset_results, config['model_name'], seed)
    summary_banner = (
        '\n' + '='*80 + '\n'
        'EVALUATION SUMMARY REPORT\n'
        f"Model: {config['model_name']} | Seed: {seed}\n"
        '='*80
    )
    print(summary_banner)
    print(summary_table)
    print('='*80 + '\n')

    with open(summary_log_path, 'a', encoding='utf-8') as stream:
        stream.write(summary_banner + '\n')
        stream.write(summary_table + '\n')
        stream.write('='*80 + '\n')

    summary_csv_path = os.path.join(out_dir, f'{log_stem}_summary.csv')
    write_summary_csv(summary_csv_path, dataset_results, config['model_name'], seed,
                      os.path.abspath(args.weights_path), ens_w)
    print(f'Consolidated log saved to: {summary_log_path}')
    print(f'Consolidated CSV saved to: {summary_csv_path}')
    print(f'All evaluation artifacts saved in: {out_dir}')


if __name__ == '__main__':
    main()
