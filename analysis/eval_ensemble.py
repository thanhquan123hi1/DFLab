"""Evaluate Decision-Level Ensemble between Fusion, CLS, and MIL heads on saved predictions.

This script operates directly on precomputed predictions (*_predictions.npz),
requiring ZERO GPU compute and ZERO retraining time. It sweeps across weights
w in [0, 1] to find the optimal balance between global scene consistency and
local artifact detection.

Usage:
    # 1. Sweep weight w from 0.0 to 1.0 in steps of 0.05 (Default)
    python analysis/eval_ensemble.py --predictions /path/to/Celeb-DF-v2_predictions.npz

    # 2. Evaluate specific weight (e.g. w = 0.5)
    python analysis/eval_ensemble.py --predictions /path/to/Celeb-DF-v2_predictions.npz --weight 0.5

    # 3. Sweep and save optimal configuration to json
    python analysis/eval_ensemble.py --predictions /path/to/Celeb-DF-v2_predictions.npz --save
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'training'))

from training.metrics.utils import get_test_metrics, format_compact_test_report, write_json


def run_ensemble_sweep(predictions_path, weights=None, mode='fusion_mil', save=False):
    path = Path(predictions_path)
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        prob = data['prob']
        label = data['label']
        names = data['image_names']
        mil_prob = data['mil_prob'] if 'mil_prob' in data else None
        cls_prob = data['cls_only_prob'] if 'cls_only_prob' in data else None

    if mil_prob is None:
        raise ValueError("File does not contain 'mil_prob'. Ensemble requires MIL branch predictions.")

    if mode == 'cls_mil' and cls_prob is not None:
        global_prob = cls_prob
        branch_name = 'CLS'
    else:
        global_prob = prob
        branch_name = 'Fusion'

    if weights is None:
        weights = np.linspace(0.0, 1.0, 21)
    elif isinstance(weights, (int, float)):
        weights = [float(weights)]

    print("=" * 82)
    print(f" DECISION FUSION / ENSEMBLE SWEEP: ({branch_name} + MIL)")
    print(f" Target File : {path.name}")
    print(f" Formula     : P_ensemble = (1 - w) * P_{branch_name} + w * P_MIL")
    print("=" * 82)
    print(f"{'Weight (w)':<12} | {'Video AUC':<12} | {'Frame AUC':<12} | {'Video EER':<12} | {'Video Acc':<12} | Note")
    print("-" * 82)

    best_v_auc = -1.0
    best_item = None
    results = []

    for w in weights:
        w_val = float(w)
        ens_prob = (1.0 - w_val) * global_prob + w_val * mil_prob
        res = get_test_metrics(ens_prob, label, names)
        v_auc = res['video_auc'] * 100
        f_auc = res['auc'] * 100
        v_eer = res['video_eer'] * 100
        v_acc = res['video_acc'] * 100

        note = ""
        if abs(w_val - 0.0) < 1e-5:
            note = f"Pure {branch_name} (w=0.0)"
        elif abs(w_val - 1.0) < 1e-5:
            note = "Pure MIL (w=1.0)"
        elif abs(w_val - 0.5) < 1e-5:
            note = "Equal (50/50)"

        print(f"w = {w_val:.2f}{'':<6} | {v_auc:6.2f}%     | {f_auc:6.2f}%     | {v_eer:6.2f}%     | {v_acc:6.2f}%    | {note}")

        item = {
            'weight': w_val,
            'video_auc': float(res['video_auc']),
            'auc': float(res['auc']),
            'video_eer': float(res['video_eer']),
            'video_acc': float(res['video_acc']),
            'metrics': res,
        }
        results.append(item)

        if res['video_auc'] > best_v_auc:
            best_v_auc = res['video_auc']
            best_item = item

    print("-" * 82)
    print(f"\n>>> OPTIMAL CONFIGURATION: w = {best_item['weight']:.2f} (Max Video AUC: {best_item['video_auc']*100:.2f}%) <<<")
    print(format_compact_test_report(f"{path.stem} [Best Ensemble w={best_item['weight']:.2f}]", best_item['metrics']))

    if save:
        out_json = path.parent / f"{path.stem}_ensemble_best_w{best_item['weight']:.2f}.json"
        write_json(out_json, best_item['metrics'])
        print(f"[SUCCESS] Saved optimal ensemble metrics to: {out_json}")

    return best_item, results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--predictions', required=True, help='Path to *_predictions.npz file')
    parser.add_argument('--weight', type=float, default=None, help='Specific MIL weight to evaluate (default: sweep [0, 1])')
    parser.add_argument('--mode', choices=['fusion_mil', 'cls_mil'], default='fusion_mil',
                        help='Which global branch to ensemble with MIL: fusion_mil or cls_mil')
    parser.add_argument('--save', action='store_true', help='Save best ensemble metrics to json')
    args = parser.parse_args()

    run_ensemble_sweep(
        predictions_path=args.predictions,
        weights=args.weight,
        mode=args.mode,
        save=args.save
    )


if __name__ == '__main__':
    main()
