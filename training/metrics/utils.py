"""Dataset-level metrics. Undefined two-class metrics are NaN, never zero."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn import metrics


def parse_metric_for_print(metric_dict):
    return '\n' + '\n'.join(f'{key}: {dict(value)}' for key, value in (metric_dict or {}).items())


def binary_metrics(y_true, y_pred):
    labels = (np.asarray(y_true).reshape(-1) != 0).astype(int)
    probs = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(labels) == 0 or len(labels) != len(probs):
        raise ValueError('Metrics require equal, nonempty label and prediction arrays')
    if not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
        raise ValueError('Predictions must be finite probabilities in [0,1]')
    pred = (probs >= .5).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    both = np.unique(labels).size == 2
    auc = eer = tpr1 = tpr5 = float('nan')
    if both:
        fpr, tpr, _ = metrics.roc_curve(labels, probs)
        auc = metrics.auc(fpr, tpr)
        eer = fpr[np.argmin(np.abs(1 - tpr - fpr))]
        tpr1, tpr5 = [float(tpr[fpr <= x].max()) for x in (.01, .05)]
    real_acc = tn / (tn + fp) if tn + fp else float('nan')
    fake_acc = tp / (tp + fn) if tp + fn else float('nan')
    ece = 0.
    for low, high in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        mask = (probs >= low) & ((probs < high) if high < 1 else (probs <= high))
        if mask.any():
            ece += mask.mean() * abs(probs[mask].mean() - labels[mask].mean())
    return dict(acc=float((pred == labels).mean()), auc=float(auc), eer=float(eer),
                ap=float(metrics.average_precision_score(labels, probs)) if both else float('nan'),
                acc_real=float(real_acc), acc_fake=float(fake_acc),
                balanced_acc=float((real_acc + fake_acc) / 2),
                f1=float(metrics.f1_score(labels, pred, zero_division=0)),
                brier=float(np.mean((probs - labels) ** 2)), ece=float(ece),
                tpr_at_fpr_01=tpr1, tpr_at_fpr_05=tpr5,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), n=len(labels))


def get_test_metrics(y_pred, y_true, img_names):
    probs = np.asarray(y_pred).reshape(-1)
    labels = (np.asarray(y_true).reshape(-1) != 0).astype(int)
    result = binary_metrics(labels, probs)
    if img_names is None or len(img_names) != len(probs):
        raise ValueError('Image paths must align exactly with evaluated predictions')
    if isinstance(img_names[0], (list, tuple)):
        video = result.copy()  # explicit video-level input
    else:
        groups = defaultdict(list)
        for index, name in enumerate(img_names):
            # Full parent path preserves dataset / method / compression identity.
            key = str(name).replace('\\', '/').rsplit('/', 1)[0]
            groups[key].append(index)
        video_labels, video_probs = [], []
        for key, indices in groups.items():
            if np.unique(labels[indices]).size != 1:
                raise ValueError(f'Inconsistent labels within video {key}')
            video_labels.append(labels[indices[0]])
            video_probs.append(probs[indices].mean())
        video = binary_metrics(video_labels, video_probs)
    result.update({f'video_{key}': value for key, value in video.items()})
    result.update(pred=probs, label=labels)
    return result


def write_json(path, data, append=False):
    """Portable JSON: undefined metrics become null; arrays remain available in NPZ."""
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k not in ('pred', 'label')}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a' if append else 'w', encoding='utf-8') as stream:
        stream.write(json.dumps(clean(data), ensure_ascii=False) + '\n')


def format_compact_test_report(name, result):
    """Format evaluation metrics into a clean, compact, human-readable summary report with a ranked leaderboard."""
    def fmt_pct(val, decimals=2):
        if val is None or not np.isfinite(val):
            return "N/A"
        return f"{val * 100:.{decimals}f}%"

    v_auc = fmt_pct(result.get('video_auc'))
    f_auc = fmt_pct(result.get('auc'))
    v_eer = fmt_pct(result.get('video_eer'))
    f_eer = fmt_pct(result.get('eer'))

    mil_v_auc = result.get('mil_video_auc')
    cls_v_auc = result.get('cls_only_video_auc', result.get('cls_video_auc'))
    ens_v_auc = result.get('ensemble_video_auc', result.get('ens_cls_mil_video_auc'))
    bdg_f_v_auc = result.get('bdg_f_mil_video_auc')
    bdg_cls_v_auc = result.get('bdg_cls_mil_video_auc')

    mil_f_auc = result.get('mil_auc')
    cls_f_auc = result.get('cls_only_auc', result.get('cls_auc'))
    ens_f_auc = result.get('ensemble_auc', result.get('ens_cls_mil_auc'))
    bdg_f_f_auc = result.get('bdg_f_mil_auc')
    bdg_cls_f_auc = result.get('bdg_cls_mil_auc')

    v_auc_extra = []
    if result.get('learned_gate'):
        v_auc_extra.append(f"Fusion: {v_auc}")
    if bdg_f_v_auc is not None and np.isfinite(bdg_f_v_auc):
        v_auc_extra.append(f"BDG(F): {fmt_pct(bdg_f_v_auc)}")
    if bdg_cls_v_auc is not None and np.isfinite(bdg_cls_v_auc) and (result.get('video_auc') != bdg_cls_v_auc):
        v_auc_extra.append(f"BDG(CLS): {fmt_pct(bdg_cls_v_auc)}")
    if ens_v_auc is not None and np.isfinite(ens_v_auc):
        v_auc_extra.append(f"Ensemble: {fmt_pct(ens_v_auc)}")
    if mil_v_auc is not None and np.isfinite(mil_v_auc):
        v_auc_extra.append(f"MIL: {fmt_pct(mil_v_auc)}")
    if cls_v_auc is not None and np.isfinite(cls_v_auc):
        v_auc_extra.append(f"CLS: {fmt_pct(cls_v_auc)}")
    v_auc_str = f"  ({ ' | '.join(v_auc_extra) })" if v_auc_extra else ""

    f_auc_extra = []
    if result.get('learned_gate'):
        f_auc_extra.append(f"Fusion: {f_auc}")
    if bdg_f_f_auc is not None and np.isfinite(bdg_f_f_auc):
        f_auc_extra.append(f"BDG(F): {fmt_pct(bdg_f_f_auc)}")
    if bdg_cls_f_auc is not None and np.isfinite(bdg_cls_f_auc) and (result.get('auc') != bdg_cls_f_auc):
        f_auc_extra.append(f"BDG(CLS): {fmt_pct(bdg_cls_f_auc)}")
    if ens_f_auc is not None and np.isfinite(ens_f_auc):
        f_auc_extra.append(f"Ensemble: {fmt_pct(ens_f_auc)}")
    if mil_f_auc is not None and np.isfinite(mil_f_auc):
        f_auc_extra.append(f"MIL: {fmt_pct(mil_f_auc)}")
    if cls_f_auc is not None and np.isfinite(cls_f_auc):
        f_auc_extra.append(f"CLS: {fmt_pct(cls_f_auc)}")
    f_auc_str = f"  ({ ' | '.join(f_auc_extra) })" if f_auc_extra else ""

    # Build Candidate Ablation Branches
    if result.get('learned_gate'):
        BRANCH_DEFS = [
            ('Learned Fusion', 'learned_fusion', ''),
            ('Feature Fusion', 'feature_fusion'),
            ('Ensemble', 'ens_cls_mil', 'ensemble'),
            ('MIL', 'mil'),
            ('CLS', 'cls_only', 'cls'),
        ]
    else:
        BRANCH_DEFS = [
            ('Adaptive BDG (CLS + MIL)', 'bdg_cls_mil'),
            ('Adaptive BDG (F + MIL)', 'bdg_f_mil'),
            ('Ensemble 50/50 (CLS + MIL)', 'ens_cls_mil', 'ensemble'),
            ('Ensemble 50/50 (F + MIL)', 'ens_f_mil'),
            ('Feature Fusion (F)', 'feature_fusion'),
            ('MIL (Patch Head only)', 'mil'),
            ('CLS only', 'cls_only', 'cls'),
        ]

    leaderboard_entries = []
    for item in BRANCH_DEFS:
        name_str = item[0]
        prefix = item[1]
        legacy_prefix = item[2] if len(item) > 2 else None

        v_auc_val = result.get(f'{prefix}_video_auc')
        if v_auc_val is None and legacy_prefix is not None:
            v_auc_val = result.get(f'{legacy_prefix}_video_auc' if legacy_prefix else 'video_auc')
        if v_auc_val is None and prefix == 'feature_fusion' and result.get('is_feat_model'):
            v_auc_val = result.get('video_auc')

        f_auc_val = result.get(f'{prefix}_auc')
        if f_auc_val is None and legacy_prefix is not None:
            f_auc_val = result.get(f'{legacy_prefix}_auc' if legacy_prefix else 'auc')
        if f_auc_val is None and prefix == 'feature_fusion' and result.get('is_feat_model'):
            f_auc_val = result.get('auc')

        if v_auc_val is None and f_auc_val is None:
            continue

        v_eer_val = result.get(f'{prefix}_video_eer')
        if v_eer_val is None and legacy_prefix is not None:
            v_eer_val = result.get(f'{legacy_prefix}_video_eer' if legacy_prefix else 'video_eer')
        if v_eer_val is None and prefix == 'feature_fusion' and result.get('is_feat_model'):
            v_eer_val = result.get('video_eer')

        f_eer_val = result.get(f'{prefix}_eer')
        if f_eer_val is None and legacy_prefix is not None:
            f_eer_val = result.get(f'{legacy_prefix}_eer' if legacy_prefix else 'eer')
        if f_eer_val is None and prefix == 'feature_fusion' and result.get('is_feat_model'):
            f_eer_val = result.get('eer')

        v_acc_val = result.get(f'{prefix}_video_acc')
        if v_acc_val is None and legacy_prefix is not None:
            v_acc_val = result.get(f'{legacy_prefix}_video_acc' if legacy_prefix else 'video_acc')
        if v_acc_val is None and prefix == 'feature_fusion' and result.get('is_feat_model'):
            v_acc_val = result.get('video_acc')

        v_ap_val = result.get(f'{prefix}_video_ap')
        if v_ap_val is None and legacy_prefix is not None:
            v_ap_val = result.get(f'{legacy_prefix}_video_ap' if legacy_prefix else 'video_ap')
        if v_ap_val is None and prefix == 'feature_fusion' and result.get('is_feat_model'):
            v_ap_val = result.get('video_ap')

        leaderboard_entries.append({
            'strategy': name_str,
            'video_auc': v_auc_val,
            'frame_auc': f_auc_val,
            'video_eer': v_eer_val,
            'frame_eer': f_eer_val,
            'video_acc': v_acc_val,
            'video_ap': v_ap_val,
        })

    def sort_score(entry):
        v = entry['video_auc']
        f = entry['frame_auc']
        v_num = float(v) if v is not None and np.isfinite(v) else -1.0
        f_num = float(f) if f is not None and np.isfinite(f) else -1.0
        return (v_num, f_num)

    leaderboard_entries.sort(key=sort_score, reverse=True)

    leaderboard_lines = []
    if len(leaderboard_entries) >= 2:
        headers = ['Rank', 'Strategy / Branch', 'Video AUC', 'Frame AUC', 'Video EER', 'Frame EER', 'Video Acc', 'Video AP']
        table_rows = []
        for rank, entry in enumerate(leaderboard_entries, 1):
            table_rows.append([
                str(rank),
                entry['strategy'],
                fmt_pct(entry['video_auc']),
                fmt_pct(entry['frame_auc']),
                fmt_pct(entry['video_eer']),
                fmt_pct(entry['frame_eer']),
                fmt_pct(entry['video_acc']),
                fmt_pct(entry['video_ap']),
            ])

        col_widths = [len(h) for h in headers]
        for row in table_rows:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(val))

        col_widths[0] = max(col_widths[0], 4)   # Rank
        col_widths[1] = max(col_widths[1], 27)  # Strategy
        for i in range(2, len(headers)):
            col_widths[i] = max(col_widths[i], 9)

        sep_line = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
        double_sep = "+=" + "=+=".join("=" * w for w in col_widths) + "=+"

        def render_tbl_row(parts):
            cells = [
                f" {parts[0]:^{col_widths[0]}} ",
                f" {parts[1]:<{col_widths[1]}} ",
            ]
            for i in range(2, len(parts)):
                cells.append(f" {parts[i]:>{col_widths[i]}} ")
            return "|" + "|".join(cells) + "|"

        leaderboard_lines = [
            "[Ablation Leaderboard - Ranked by Video AUC]:",
            sep_line,
            render_tbl_row(headers),
            double_sep,
        ]
        for row in table_rows:
            leaderboard_lines.append(render_tbl_row(row))
        leaderboard_lines.append(sep_line)

    v_acc = fmt_pct(result.get('video_acc'))
    f_acc = fmt_pct(result.get('acc'))
    v_real = fmt_pct(result.get('video_acc_real'))
    v_fake = fmt_pct(result.get('video_acc_fake'))

    v_ap = fmt_pct(result.get('video_ap'))
    f_ap = fmt_pct(result.get('ap'))

    v_tpr1 = fmt_pct(result.get('video_tpr_at_fpr_01'))
    v_tpr5 = fmt_pct(result.get('video_tpr_at_fpr_05'))

    tn = result.get('video_tn', 'N/A')
    fp = result.get('video_fp', 'N/A')
    fn = result.get('video_fn', 'N/A')
    tp = result.get('video_tp', 'N/A')

    v_n = result.get('video_n', 'N/A')
    f_n = result.get('n', 'N/A')

    details_hdr = "[Secondary Details - Adaptive BDG]:" if (result.get('gating_w_mean') is not None and not leaderboard_lines) else "[Secondary Details]:"
    if result.get('learned_gate'):
        details_hdr = "[Secondary Details - Fusion (learned gate)]:"

    secondary = [
        details_hdr,
        f"  - Acc (Video/Frame)  : {v_acc} / {f_acc} (Real: {v_real}, Fake: {v_fake})",
        f"  - AP  (Video/Frame)  : {v_ap} / {f_ap}",
        f"  - Low FPR Detection  : TPR@1% = {v_tpr1} | TPR@5% = {v_tpr5}",
    ]
    gw_mean = result.get('gating_w_mean')
    gw_f_mean = result.get('gating_w_f_mean')
    if gw_mean is not None:
        gw_real = result.get('gating_w_real')
        gw_fake = result.get('gating_w_fake')
        real_str = f"{gw_real:.2f}" if gw_real is not None and np.isfinite(gw_real) else "N/A"
        fake_str = f"{gw_fake:.2f}" if gw_fake is not None and np.isfinite(gw_fake) else "N/A"
        cls_lbl = "Gating Behavior (CLS)" if gw_f_mean is not None else "Gating Behavior    "
        secondary.append(f"  - {cls_lbl}: Mean w = {gw_mean:.2f} (Real: {real_str}, Fake: {fake_str})")
    if gw_f_mean is not None:
        gw_f_real = result.get('gating_w_f_real')
        gw_f_fake = result.get('gating_w_f_fake')
        f_real_str = f"{gw_f_real:.2f}" if gw_f_real is not None and np.isfinite(gw_f_real) else "N/A"
        f_fake_str = f"{gw_f_fake:.2f}" if gw_f_fake is not None and np.isfinite(gw_f_fake) else "N/A"
        secondary.append(f"  - Gating Behavior (F)  : Mean w = {gw_f_mean:.2f} (Real: {f_real_str}, Fake: {f_fake_str})")
    secondary.extend([
        f"  - Video Confusion    : TN={tn}, FP={fp}, FN={fn}, TP={tp}",
        "-" * 70,
    ])

    lines = [
        f"\n>>> [{name}] TEST SUMMARY ({v_n} videos, {f_n} frames) <<<",
        "-" * 70,
        f"* VIDEO AUC : {v_auc}{v_auc_str}",
        f"* FRAME AUC : {f_auc}{f_auc_str}",
        f"* VIDEO EER : {v_eer}  (Frame EER: {f_eer})",
        "-" * 70,
    ]
    if leaderboard_lines:
        lines.extend(leaderboard_lines)
        lines.append("-" * 70)
    lines.extend(secondary)
    return "\n".join(lines)
