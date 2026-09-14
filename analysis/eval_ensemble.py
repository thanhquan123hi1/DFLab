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


def plot_ensemble_curves(all_results, plot_path, branch_name='CLS'):
    """Plot Video AUC vs MIL weight curves for all evaluated datasets."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6), dpi=150)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']

    for idx, (ds_name, (best_item, results)) in enumerate(all_results.items()):
        ws = [r['weight'] for r in results]
        v_aucs = [r['video_auc'] * 100 for r in results]
        color = colors[idx % len(colors)]
        label = f"{ds_name} (Max: {best_item['video_auc']*100:.2f}% @ w={best_item['weight']:.2f})"
        plt.plot(ws, v_aucs, marker='o', markersize=4, linewidth=2, color=color, label=label)
        plt.scatter([best_item['weight']], [best_item['video_auc'] * 100],
                    color=color, s=90, edgecolors='black', linewidth=1.2, zorder=5)

    if len(all_results) > 1:
        # Calculate joint average curve
        first_res = next(iter(all_results.values()))[1]
        ws = [r['weight'] for r in first_res]
        avg_aucs = []
        for i in range(len(ws)):
            aucs = [res_list[1][i]['video_auc'] * 100 for res_list in all_results.values()]
            avg_aucs.append(float(np.mean(aucs)))
        best_avg_idx = int(np.argmax(avg_aucs))
        best_avg_w = ws[best_avg_idx]
        best_avg_auc = avg_aucs[best_avg_idx]
        plt.plot(ws, avg_aucs, marker='s', markersize=5, linewidth=2.5, linestyle='--',
                 color='#000000', label=f"Average (Peak: {best_avg_auc:.2f}% @ w={best_avg_w:.2f})")
        plt.scatter([best_avg_w], [best_avg_auc], color='black', s=110, edgecolors='gold', linewidth=1.5, zorder=6)

    plt.title(f"Decision Ensemble Sweep ({branch_name} + MIL): Video AUC vs Weight (w)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel(f"MIL Weight (w)  [w = 0.0: Pure {branch_name}  |  w = 1.0: Pure MIL]", fontsize=11)
    plt.ylabel("Video AUC (%)", fontsize=11)
    plt.xticks(np.linspace(0.0, 1.0, 11))
    plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9, shadow=True)
    plt.tight_layout()

    out_file = Path(plot_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file)
    plt.close()
    print(f"\n[PLOT SUCCESS] Saved ensemble curve to: {out_file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--predictions', nargs='+', required=True, help='Path to one or more *_predictions.npz files')
    parser.add_argument('--weight', type=float, default=None, help='Specific MIL weight to evaluate (default: sweep [0, 1])')
    parser.add_argument('--mode', choices=['fusion_mil', 'cls_mil'], default='cls_mil',
                        help='Which global branch to ensemble with MIL: cls_mil (recommended) or fusion_mil')
    parser.add_argument('--save', action='store_true', help='Save best ensemble metrics to json')
    parser.add_argument('--plot', action='store_true', help='Generate and save AUC vs weight curve plot')
    parser.add_argument('--plot_out', default=None, help='Custom path to save the plot image')
    args = parser.parse_args()

    all_results = {}
    for pred_path in args.predictions:
        p = Path(pred_path)
        name = p.stem.replace('_predictions', '')
        best_item, results = run_ensemble_sweep(
            predictions_path=p,
            weights=args.weight,
            mode=args.mode,
            save=args.save
        )
        all_results[name] = (best_item, results)

    if len(all_results) > 1 and args.weight is None:
        print("\n" + "=" * 82)
        print(f" JOINT MULTI-DATASET SUMMARY ({args.mode.upper()})")
        print("=" * 82)
        datasets = list(all_results.keys())
        header = f"{'Weight (w)':<12} | " + " | ".join(f"{d[:14]:<14}" for d in datasets) + " | Average AUC"
        print(header)
        print("-" * len(header))

        first_res = next(iter(all_results.values()))[1]
        ws = [r['weight'] for r in first_res]
        best_joint_w = None
        best_joint_avg = -1.0

        for i, w in enumerate(ws):
            aucs = [all_results[d][1][i]['video_auc'] * 100 for d in datasets]
            avg_auc = float(np.mean(aucs))
            auc_str = " | ".join(f"{a:6.2f}%{'':<7}" for a in aucs)
            print(f"w = {w:.2f}{'':<6} | {auc_str} | {avg_auc:6.2f}%")
            if avg_auc > best_joint_avg:
                best_joint_avg = avg_auc
                best_joint_w = w

        print("-" * len(header))
        print(f">>> JOINT OPTIMAL WEIGHT: w = {best_joint_w:.2f} (Average Video AUC: {best_joint_avg:.2f}%) <<<\n")

    if args.plot or args.plot_out:
        branch_name = 'CLS' if args.mode == 'cls_mil' else 'Fusion'
        plot_path = args.plot_out
        if plot_path is None:
            first_path = Path(args.predictions[0])
            plot_path = first_path.parent / f"ensemble_{args.mode}_curve.png"
        plot_ensemble_curves(all_results, plot_path, branch_name=branch_name)


if __name__ == '__main__':
    main()
