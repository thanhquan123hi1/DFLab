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
    """Format evaluation metrics into a clean, compact, human-readable summary report."""
    def fmt_pct(val, decimals=2):
        if val is None or not np.isfinite(val):
            return "N/A"
        return f"{val * 100:.{decimals}f}%"

    v_auc = fmt_pct(result.get('video_auc'))
    f_auc = fmt_pct(result.get('auc'))
    v_eer = fmt_pct(result.get('video_eer'))
    f_eer = fmt_pct(result.get('eer'))

    mil_v_auc = result.get('mil_video_auc')
    cls_v_auc = result.get('cls_only_video_auc')
    ens_v_auc = result.get('ensemble_video_auc')
    mil_f_auc = result.get('mil_auc')
    cls_f_auc = result.get('cls_only_auc')
    ens_f_auc = result.get('ensemble_auc')

    v_auc_extra = []
    if ens_v_auc is not None and np.isfinite(ens_v_auc):
        v_auc_extra.append(f"Ensemble: {fmt_pct(ens_v_auc)}")
    if mil_v_auc is not None and np.isfinite(mil_v_auc):
        v_auc_extra.append(f"MIL: {fmt_pct(mil_v_auc)}")
    if cls_v_auc is not None and np.isfinite(cls_v_auc):
        v_auc_extra.append(f"CLS: {fmt_pct(cls_v_auc)}")
    v_auc_str = f"  ({ ' | '.join(v_auc_extra) })" if v_auc_extra else ""

    f_auc_extra = []
    if ens_f_auc is not None and np.isfinite(ens_f_auc):
        f_auc_extra.append(f"Ensemble: {fmt_pct(ens_f_auc)}")
    if mil_f_auc is not None and np.isfinite(mil_f_auc):
        f_auc_extra.append(f"MIL: {fmt_pct(mil_f_auc)}")
    if cls_f_auc is not None and np.isfinite(cls_f_auc):
        f_auc_extra.append(f"CLS: {fmt_pct(cls_f_auc)}")
    f_auc_str = f"  ({ ' | '.join(f_auc_extra) })" if f_auc_extra else ""

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

    lines = [
        f"\n>>> [{name}] TEST SUMMARY ({v_n} videos, {f_n} frames) <<<",
        "-" * 70,
        f"* VIDEO AUC : {v_auc}{v_auc_str}",
        f"* FRAME AUC : {f_auc}{f_auc_str}",
        f"* VIDEO EER : {v_eer}  (Frame EER: {f_eer})",
        "-" * 70,
        "[Secondary Details - Fusion]:",
        f"  - Acc (Video/Frame)  : {v_acc} / {f_acc} (Real: {v_real}, Fake: {v_fake})",
        f"  - AP  (Video/Frame)  : {v_ap} / {f_ap}",
        f"  - Low FPR Detection  : TPR@1% = {v_tpr1} | TPR@5% = {v_tpr5}",
        f"  - Video Confusion    : TN={tn}, FP={fp}, FN={fn}, TP={tp}",
        "-" * 70,
    ]
    return "\n".join(lines)
