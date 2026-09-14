"""Portable run locations and lossless tabular evaluation reports."""
import csv
import math
import os
import re
from pathlib import Path


def safe_name(value):
    return re.sub(r'[<>:"/\\|?*\s]+', '_', str(value)).strip(' .') or 'unknown'


def evaluation_directory(weights_path, checkpoint, log_root, output_dir=None, fallback_dir=None):
    if output_dir:
        return Path(output_dir)
    parent = Path(weights_path).resolve().parent
    # Standard best checkpoint: <run>/validation/<dataset>/ckpt_best.pth.
    run = parent.parent.parent if parent.parent.name == 'validation' else parent
    run_name = checkpoint.get('run_name') or run.name

    if log_root:
        if fallback_dir is not None and not os.path.exists(str(log_root)):
            return Path(fallback_dir)
        return Path(log_root) / safe_name(run_name) / 'evaluation'

    if fallback_dir is not None and (not run.exists() or parent.parent.name != 'validation'):
        return Path(fallback_dir)

    return run / 'evaluation'


def compact_training_metrics(values):
    keys = ('overall', 'loss_ce', 'loss_mil', 'auc', 'video_auc', 'eer',
            'fusion_alpha', 'gating_weight_mean', 'lr')
    return ' | '.join(f'{key}={values[key]:.6g}' for key in keys if key in values)


METRIC_NAMES = ('auc', 'eer', 'ap', 'acc', 'acc_real', 'acc_fake', 'balanced_acc',
                'f1', 'brier', 'ece', 'tpr_at_fpr_01', 'tpr_at_fpr_05',
                'tn', 'fp', 'fn', 'tp', 'n')
FEATURE_MODELS = ('bias_sspanet_feat_mil', 'bias_sspanet_ff_mil', 'camil', 'bias_saf_mil', 'saf_mil')


def _get_metric_rows(result, model, seed, dataset, checkpoint, ensemble_weight):
    if result.get('learned_gate'):
        branches = [
            ('', 'learned_fusion'),
            ('feature_fusion_', 'feature_fusion'),
            ('ens_f_mil_', 'ens_f_mil'),
            ('ens_cls_mil_', 'ensemble'),
            ('mil_', 'mil'),
            ('cls_only_', 'cls'),
        ]
        rows = []
        for prefix, branch in branches:
            for level in ('frame', 'video'):
                key_prefix = prefix + ('video_' if level == 'video' else '')
                if key_prefix + 'auc' not in result:
                    # Fallback for ensemble_ prefix
                    if branch == 'ensemble':
                        fallback_prefix = f'ensemble_{"video_" if level == "video" else ""}'
                        if fallback_prefix + 'auc' in result:
                            key_prefix = fallback_prefix
                        else:
                            continue
                    else:
                        continue
                row = dict(model=model, seed=seed, dataset=dataset, checkpoint=checkpoint,
                           branch=branch, level=level,
                           ensemble_weight=ensemble_weight if branch == 'ensemble' else '')
                for name in METRIC_NAMES:
                    value = result.get(key_prefix + name)
                    row[name] = '' if value is None or not math.isfinite(float(value)) else value
                rows.append(row)
        return rows

    # Check if 7-branch ablation metrics exist in result
    has_ablation_branches = any(
        (f'{b}_auc' in result or f'{b}_video_auc' in result)
        for b in ('bdg_cls_mil', 'bdg_f_mil', 'ens_cls_mil', 'ens_f_mil', 'feature_fusion')
    )
    if has_ablation_branches:
        branches = [
            ('bdg_cls_mil_', 'bdg_cls_mil'),
            ('bdg_f_mil_', 'bdg_f_mil'),
            ('ens_cls_mil_', 'ens_cls_mil'),
            ('ens_f_mil_', 'ens_f_mil'),
            ('feature_fusion_', 'feature_fusion'),
            ('mil_', 'mil'),
            ('cls_only_', 'cls'),
        ]
        rows = []
        for prefix, branch in branches:
            for level in ('frame', 'video'):
                key_prefix = prefix + ('video_' if level == 'video' else '')
                if key_prefix + 'auc' not in result:
                    # Fallback for feature_fusion if stored as unprefixed primary
                    if branch == 'feature_fusion' and (model in FEATURE_MODELS or result.get('is_feat_model')):
                        key_prefix = 'video_' if level == 'video' else ''
                        if key_prefix + 'auc' not in result:
                            continue
                    else:
                        continue
                is_ensemble = branch.startswith('ens_') or branch == 'ensemble'
                row = dict(model=model, seed=seed, dataset=dataset, checkpoint=checkpoint,
                           branch=branch, level=level,
                           ensemble_weight=ensemble_weight if is_ensemble else '')
                for name in METRIC_NAMES:
                    value = result.get(key_prefix + name)
                    row[name] = '' if value is None or not math.isfinite(float(value)) else value
                rows.append(row)
        return rows

    # Legacy fallback behavior
    primary = ('learned_fusion' if result.get('learned_gate') else
               'adaptive_bdg' if 'gating_w_mean' in result else
               'fusion' if model in FEATURE_MODELS else 'primary')
    branches = [('', primary), ('cls_only_', 'cls'), ('mil_', 'mil'),
                ('ensemble_', 'ensemble')]
    rows = []
    for prefix, branch in branches:
        for level in ('frame', 'video'):
            key_prefix = prefix + ('video_' if level == 'video' else '')
            if key_prefix + 'auc' not in result:
                continue
            row = dict(model=model, seed=seed, dataset=dataset, checkpoint=checkpoint,
                       branch=branch, level=level,
                       ensemble_weight=ensemble_weight if branch == 'ensemble' else '')
            for name in METRIC_NAMES:
                value = result.get(key_prefix + name)
                row[name] = '' if value is None or not math.isfinite(float(value)) else value
            rows.append(row)
    return rows


def write_metrics_csv(path, result, model, seed, dataset, checkpoint, ensemble_weight):
    """One row per branch and level; probability metrics remain on the 0–1 scale."""
    fields = ['model', 'seed', 'dataset', 'checkpoint', 'branch', 'level',
              'ensemble_weight', *METRIC_NAMES]
    rows = _get_metric_rows(result, model, seed, dataset, checkpoint, ensemble_weight)
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_csv(path, dataset_results, model, seed, checkpoint, ensemble_weight):
    """Consolidated CSV of all evaluated datasets with AVERAGE rows across datasets."""
    fields = ['model', 'seed', 'dataset', 'checkpoint', 'branch', 'level',
              'ensemble_weight', *METRIC_NAMES]
    all_rows = []
    for dataset_name, result in dataset_results:
        all_rows.extend(_get_metric_rows(result, model, seed, dataset_name, checkpoint, ensemble_weight))

    # Identify unique (branch, level) combinations in order
    branch_levels = []
    for r in all_rows:
        bl = (r['branch'], r['level'])
        if bl not in branch_levels:
            branch_levels.append(bl)

    avg_rows = []
    for branch, level in branch_levels:
        matching = [r for r in all_rows if r['branch'] == branch and r['level'] == level]
        is_ensemble = branch.startswith('ens_') or branch == 'ensemble'
        avg_row = dict(model=model, seed=seed, dataset='AVERAGE', checkpoint=checkpoint,
                       branch=branch, level=level,
                       ensemble_weight=ensemble_weight if is_ensemble else '')
        for name in METRIC_NAMES:
            valid_vals = [float(r[name]) for r in matching if r[name] != '' and r[name] is not None]
            if valid_vals:
                avg_row[name] = sum(valid_vals) / len(valid_vals)
            else:
                avg_row[name] = ''
        avg_rows.append(avg_row)

    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
        for row in avg_rows:
            writer.writerow(row)


def format_summary_table(dataset_results, model_name=None, seed=None):
    """Generate an aligned ASCII summary table comparing all datasets with an AVERAGE row."""
    if not dataset_results:
        return ""

    has_video_auc = any('video_auc' in res for _, res in dataset_results)
    has_video_eer = any('video_eer' in res for _, res in dataset_results)
    has_ensemble = any(('ensemble_auc' in res or 'ensemble_video_auc' in res) for _, res in dataset_results)

    headers = ['Dataset', 'Frame AUC']
    if has_video_auc:
        headers.append('Video AUC')
    headers.append('Frame EER')
    if has_video_eer:
        headers.append('Video EER')
    headers.append('Frame Acc')
    if has_ensemble:
        headers.append('Ens Frame AUC')
        if has_video_auc:
            headers.append('Ens Video AUC')

    def fmt_val(v):
        if v is None:
            return '-'
        try:
            fv = float(v)
            if math.isnan(fv) or not math.isfinite(fv):
                return '-'
            return f"{fv:.4f}"
        except (ValueError, TypeError):
            return '-'

    rows = []
    for name, res in dataset_results:
        row = [name, fmt_val(res.get('auc'))]
        if has_video_auc:
            row.append(fmt_val(res.get('video_auc')))
        row.append(fmt_val(res.get('eer')))
        if has_video_eer:
            row.append(fmt_val(res.get('video_eer')))
        row.append(fmt_val(res.get('acc')))
        if has_ensemble:
            row.append(fmt_val(res.get('ensemble_auc')))
            if has_video_auc:
                row.append(fmt_val(res.get('ensemble_video_auc')))
        rows.append(row)

    # Calculate AVERAGE row
    avg_row = ['AVERAGE']
    for col_idx in range(1, len(headers)):
        valid_vals = []
        for r in rows:
            v = r[col_idx]
            if v != '-':
                try:
                    valid_vals.append(float(v))
                except ValueError:
                    pass
        if valid_vals:
            avg_row.append(f"{sum(valid_vals) / len(valid_vals):.4f}")
        else:
            avg_row.append('-')

    all_table_rows = [headers] + rows + [avg_row]
    col_widths = [max(len(str(r[i])) for r in all_table_rows) for i in range(len(headers))]
    col_widths = [max(w, 10) for w in col_widths]
    col_widths[0] = max(col_widths[0], 12)

    sep_line = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    double_sep = "+=" + "=+=".join("=" * w for w in col_widths) + "=+"

    def render_row(vals):
        parts = [f" {vals[0]:<{col_widths[0]}} "]
        for i in range(1, len(vals)):
            parts.append(f" {vals[i]:>{col_widths[i]}} ")
        return "|" + "|".join(parts) + "|"

    lines = [
        sep_line,
        render_row(headers),
        double_sep
    ]
    for r in rows:
        lines.append(render_row(r))
    lines.append(sep_line)
    lines.append(render_row(avg_row))
    lines.append(sep_line)

    primary_table = "\n".join(lines)

    # Check for ablation branches across datasets
    has_ablation = any(
        any((f'{b}_auc' in res or f'{b}_video_auc' in res)
            for b in ('bdg_cls_mil', 'bdg_f_mil', 'ens_cls_mil', 'ens_f_mil', 'feature_fusion'))
        for _, res in dataset_results
    )
    if not has_ablation:
        return primary_table

    # Format multi-branch ablation summary leaderboard across datasets
    def fmt_pct(val, decimals=2):
        if val is None or not (isinstance(val, (int, float)) and math.isfinite(val)):
            return "N/A"
        return f"{float(val) * 100:.{decimals}f}%"

    branch_specs = [
        ('Adaptive BDG (CLS + MIL)', 'bdg_cls_mil'),
        ('Adaptive BDG (F + MIL)', 'bdg_f_mil'),
        ('Ensemble 50/50 (CLS + MIL)', 'ens_cls_mil'),
        ('Ensemble 50/50 (F + MIL)', 'ens_f_mil'),
        ('Feature Fusion (F)', 'feature_fusion'),
        ('MIL (Patch Head only)', 'mil'),
        ('CLS only', 'cls_only'),
    ]

    ablation_entries = []
    for strat_name, prefix in branch_specs:
        v_aucs, f_aucs, v_eers, f_eers, v_accs, v_aps = [], [], [], [], [], []
        for _, res in dataset_results:
            v_auc = res.get(f'{prefix}_video_auc')
            if v_auc is None and prefix == 'feature_fusion' and (model_name in FEATURE_MODELS or res.get('is_feat_model')):
                v_auc = res.get('video_auc')
            f_auc = res.get(f'{prefix}_auc')
            if f_auc is None and prefix == 'feature_fusion' and (model_name in FEATURE_MODELS or res.get('is_feat_model')):
                f_auc = res.get('auc')

            if v_auc is not None and math.isfinite(float(v_auc)):
                v_aucs.append(float(v_auc))
            if f_auc is not None and math.isfinite(float(f_auc)):
                f_aucs.append(float(f_auc))

            v_eer = res.get(f'{prefix}_video_eer')
            if v_eer is None and prefix == 'feature_fusion' and (model_name in FEATURE_MODELS or res.get('is_feat_model')):
                v_eer = res.get('video_eer')
            f_eer = res.get(f'{prefix}_eer')
            if f_eer is None and prefix == 'feature_fusion' and (model_name in FEATURE_MODELS or res.get('is_feat_model')):
                f_eer = res.get('eer')
            if v_eer is not None and math.isfinite(float(v_eer)):
                v_eers.append(float(v_eer))
            if f_eer is not None and math.isfinite(float(f_eer)):
                f_eers.append(float(f_eer))

            v_acc = res.get(f'{prefix}_video_acc')
            if v_acc is None and prefix == 'feature_fusion' and (model_name in FEATURE_MODELS or res.get('is_feat_model')):
                v_acc = res.get('video_acc')
            if v_acc is not None and math.isfinite(float(v_acc)):
                v_accs.append(float(v_acc))

            v_ap = res.get(f'{prefix}_video_ap')
            if v_ap is None and prefix == 'feature_fusion' and (model_name in FEATURE_MODELS or res.get('is_feat_model')):
                v_ap = res.get('video_ap')
            if v_ap is not None and math.isfinite(float(v_ap)):
                v_aps.append(float(v_ap))

        if not v_aucs and not f_aucs:
            continue

        ablation_entries.append({
            'strategy': strat_name,
            'video_auc': sum(v_aucs) / len(v_aucs) if v_aucs else None,
            'frame_auc': sum(f_aucs) / len(f_aucs) if f_aucs else None,
            'video_eer': sum(v_eers) / len(v_eers) if v_eers else None,
            'frame_eer': sum(f_eers) / len(f_eers) if f_eers else None,
            'video_acc': sum(v_accs) / len(v_accs) if v_accs else None,
            'video_ap': sum(v_aps) / len(v_aps) if v_aps else None,
        })

    if len(ablation_entries) < 2:
        return primary_table

    def sort_score(entry):
        v = entry['video_auc']
        f = entry['frame_auc']
        v_num = float(v) if v is not None and math.isfinite(v) else -1.0
        f_num = float(f) if f is not None and math.isfinite(f) else -1.0
        return (v_num, f_num)

    ablation_entries.sort(key=sort_score, reverse=True)

    abl_headers = ['Rank', 'Strategy / Branch', 'Video AUC', 'Frame AUC', 'Video EER', 'Frame EER', 'Video Acc', 'Video AP']
    abl_rows = []
    for rank, entry in enumerate(ablation_entries, 1):
        abl_rows.append([
            str(rank),
            entry['strategy'],
            fmt_pct(entry['video_auc']),
            fmt_pct(entry['frame_auc']),
            fmt_pct(entry['video_eer']),
            fmt_pct(entry['frame_eer']),
            fmt_pct(entry['video_acc']),
            fmt_pct(entry['video_ap']),
        ])

    abl_col_widths = [len(h) for h in abl_headers]
    for row in abl_rows:
        for i, val in enumerate(row):
            abl_col_widths[i] = max(abl_col_widths[i], len(val))
    abl_col_widths[0] = max(abl_col_widths[0], 4)
    abl_col_widths[1] = max(abl_col_widths[1], 27)
    for i in range(2, len(abl_headers)):
        abl_col_widths[i] = max(abl_col_widths[i], 9)

    abl_sep = "+-" + "-+-".join("-" * w for w in abl_col_widths) + "-+"
    abl_double_sep = "+=" + "=+=".join("=" * w for w in abl_col_widths) + "=+"

    def render_abl_row(parts):
        cells = [
            f" {parts[0]:^{abl_col_widths[0]}} ",
            f" {parts[1]:<{abl_col_widths[1]}} ",
        ]
        for i in range(2, len(parts)):
            cells.append(f" {parts[i]:>{abl_col_widths[i]}} ")
        return "|" + "|".join(cells) + "|"

    num_ds = len(dataset_results)
    ds_label = f"Average Across {num_ds} Datasets" if num_ds > 1 else f"Dataset: {dataset_results[0][0]}"
    abl_lines = [
        f"[Ablation Leaderboard Summary - {ds_label} (Ranked by Video AUC)]:",
        abl_sep,
        render_abl_row(abl_headers),
        abl_double_sep,
    ]
    for row in abl_rows:
        abl_lines.append(render_abl_row(row))
    abl_lines.append(abl_sep)

    return primary_table + "\n\n" + "\n".join(abl_lines)

