import csv
import pytest
from metrics.reporting import (
    evaluation_directory,
    write_metrics_csv,
    write_summary_csv,
    format_summary_table,
)


def test_evaluation_directory_uses_training_root_and_run(tmp_path):
    weights = tmp_path / 'old' / 'run_42' / 'validation' / 'source' / 'ckpt_best.pth'
    root = tmp_path / 'configured_logs'
    assert evaluation_directory(weights, {}, root) == root / 'run_42' / 'evaluation'
    assert evaluation_directory(weights, {'run_name': 'saved_run'}, root) == root / 'saved_run' / 'evaluation'
    assert evaluation_directory(weights, {}, root, tmp_path / 'explicit') == tmp_path / 'explicit'
    assert evaluation_directory(weights, {}, None) == tmp_path / 'old' / 'run_42' / 'evaluation'
    assert evaluation_directory(tmp_path / 'run_42' / 'ckpt_last.pth', {}, root) == root / 'run_42' / 'evaluation'


def test_evaluation_directory_fallback(tmp_path):
    weights = tmp_path / 'weights' / 'best.pth'
    non_existent_root = '/content/drive/MyDrive/NCKH'
    fallback = tmp_path / 'evaluations' / 'eval_camil_1024'

    # When log_root does not exist on disk and fallback_dir is provided
    result = evaluation_directory(weights, {}, non_existent_root, fallback_dir=fallback)
    assert result == fallback

    # When log_root exists on disk, use configured log_root
    existing_root = tmp_path / 'existing_logs'
    existing_root.mkdir()
    result_existing = evaluation_directory(weights, {'run_name': 'camil_run'}, existing_root, fallback_dir=fallback)
    assert result_existing == existing_root / 'camil_run' / 'evaluation'


def test_csv_preserves_precision_and_branch_identity(tmp_path):
    result = dict(auc=.9532881692002644, video_auc=.96, n=32, video_n=2,
                  cls_only_auc=.95, mil_auc=.94, ensemble_auc=.97, eer=float('nan'))
    path = tmp_path / 'metrics.csv'
    write_metrics_csv(path, result, 'bias_sspanet_feat_mil', 42, 'Celeb-DF-v2', 'best.pth', .5)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert [(r['branch'], r['level']) for r in rows] == [
        ('fusion', 'frame'), ('fusion', 'video'), ('cls', 'frame'),
        ('mil', 'frame'), ('ensemble', 'frame')]
    assert float(rows[0]['auc']) == result['auc']
    assert rows[0]['eer'] == ''
    assert rows[0]['seed'] == '42'
    assert rows[-1]['ensemble_weight'] == '0.5'


def test_write_summary_csv_and_average(tmp_path):
    res1 = dict(auc=0.90, video_auc=0.92, acc=0.85, eer=0.10)
    res2 = dict(auc=0.80, video_auc=0.84, acc=0.75, eer=0.20)
    dataset_results = [('DatasetA', res1), ('DatasetB', res2)]
    summary_path = tmp_path / 'summary.csv'

    write_summary_csv(summary_path, dataset_results, 'camil', 1024, 'model.pth', None)

    with summary_path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))

    datasets_in_csv = [r['dataset'] for r in rows]
    assert 'DatasetA' in datasets_in_csv
    assert 'DatasetB' in datasets_in_csv
    assert 'AVERAGE' in datasets_in_csv

    # Check that average calculation is correct for fusion frame
    avg_frame = next(r for r in rows if r['dataset'] == 'AVERAGE' and r['level'] == 'frame')
    assert pytest.approx(float(avg_frame['auc']), 1e-4) == 0.85
    assert pytest.approx(float(avg_frame['acc']), 1e-4) == 0.80
    assert pytest.approx(float(avg_frame['eer']), 1e-4) == 0.15


def test_format_summary_table():
    res1 = dict(auc=0.985, video_auc=0.9912, acc=0.945, eer=0.045)
    res2 = dict(auc=0.962, video_auc=0.9750, acc=0.912, eer=0.082)
    dataset_results = [('FF-DF', res1), ('FF-F2F', res2)]

    table_str = format_summary_table(dataset_results, 'camil', 1024)
    assert 'Dataset' in table_str
    assert 'FF-DF' in table_str
    assert 'FF-F2F' in table_str
    assert 'AVERAGE' in table_str
    assert '0.9735' in table_str  # Average of 0.985 and 0.962


def test_reporting_seven_branches_csv_and_summary(tmp_path):
    res1 = {
        'bdg_cls_mil_auc': 0.92, 'bdg_cls_mil_video_auc': 0.94, 'bdg_cls_mil_eer': 0.08,
        'bdg_f_mil_auc': 0.95, 'bdg_f_mil_video_auc': 0.97, 'bdg_f_mil_eer': 0.05,
        'ens_cls_mil_auc': 0.91, 'ens_cls_mil_video_auc': 0.93, 'ens_cls_mil_eer': 0.09,
        'ens_f_mil_auc': 0.94, 'ens_f_mil_video_auc': 0.96, 'ens_f_mil_eer': 0.06,
        'feature_fusion_auc': 0.90, 'feature_fusion_video_auc': 0.92, 'feature_fusion_eer': 0.10,
        'mil_auc': 0.88, 'mil_video_auc': 0.89, 'mil_eer': 0.12,
        'cls_only_auc': 0.86, 'cls_only_video_auc': 0.87, 'cls_only_eer': 0.14,
    }
    csv_path = tmp_path / 'metrics_7.csv'
    write_metrics_csv(csv_path, res1, 'bias_sspanet_feat_mil', 42, 'Celeb-DF-v2', 'best.pth', 0.5)

    with csv_path.open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 14
    expected_order = [
        ('bdg_cls_mil', 'frame'), ('bdg_cls_mil', 'video'),
        ('bdg_f_mil', 'frame'), ('bdg_f_mil', 'video'),
        ('ens_cls_mil', 'frame'), ('ens_cls_mil', 'video'),
        ('ens_f_mil', 'frame'), ('ens_f_mil', 'video'),
        ('feature_fusion', 'frame'), ('feature_fusion', 'video'),
        ('mil', 'frame'), ('mil', 'video'),
        ('cls', 'frame'), ('cls', 'video'),
    ]
    assert [(r['branch'], r['level']) for r in rows] == expected_order
    assert float(rows[2]['auc']) == 0.95  # bdg_f_mil frame
    assert float(rows[3]['auc']) == 0.97  # bdg_f_mil video

    # Summary with multiple datasets
    res2 = {k: v - 0.1 for k, v in res1.items()}
    dataset_results = [('Dataset1', res1), ('Dataset2', res2)]
    sum_path = tmp_path / 'summary_7.csv'
    write_summary_csv(sum_path, dataset_results, 'bias_sspanet_feat_mil', 42, 'best.pth', 0.5)

    with sum_path.open(encoding='utf-8-sig') as f:
        sum_rows = list(csv.DictReader(f))

    assert len(sum_rows) == 42  # 14 for D1 + 14 for D2 + 14 for AVERAGE
    avg_bdg_f_video = next(r for r in sum_rows if r['dataset'] == 'AVERAGE' and r['branch'] == 'bdg_f_mil' and r['level'] == 'video')
    assert pytest.approx(float(avg_bdg_f_video['auc']), 1e-4) == 0.92  # Avg of 0.97 and 0.87


def test_ln_sspanet_mil_reporting_excludes_feature_fusion(tmp_path):
    from metrics.utils import format_compact_test_report
    # LN-SSPANet-MIL has BDG(CLS+MIL), Ensemble(CLS+MIL), MIL, CLS, but NO feature fusion
    res = {
        'video_auc': 0.96, 'auc': 0.95, 'video_eer': 0.08, 'eer': 0.09, 'acc': 0.91,
        'bdg_cls_mil_auc': 0.96, 'bdg_cls_mil_video_auc': 0.96, 'bdg_cls_mil_eer': 0.08,
        'ens_cls_mil_auc': 0.92, 'ens_cls_mil_video_auc': 0.92, 'ens_cls_mil_eer': 0.11,
        'mil_auc': 0.88, 'mil_video_auc': 0.88, 'mil_eer': 0.14,
        'cls_only_auc': 0.85, 'cls_only_video_auc': 0.85, 'cls_only_eer': 0.16,
    }
    # CSV rows must NOT include feature_fusion
    csv_path = tmp_path / 'ln_metrics.csv'
    write_metrics_csv(csv_path, res, 'ln_sspanet_mil', 1024, 'Celeb-DF-v2', 'best.pth', 0.5)
    with csv_path.open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    branches = [r['branch'] for r in rows]
    assert 'feature_fusion' not in branches
    assert 'bdg_f_mil' not in branches
    assert 'ens_f_mil' not in branches
    assert 'bdg_cls_mil' in branches
    assert 'ens_cls_mil' in branches
    assert 'mil' in branches
    assert 'cls' in branches
    assert len(rows) == 8  # 4 branches x 2 levels

    # Leaderboard report must NOT include Feature Fusion (F)
    report = format_compact_test_report('Celeb-DF-v2', res)
    assert 'Feature Fusion (F)' not in report
    assert 'Adaptive BDG (CLS + MIL)' in report
    assert 'Ensemble 50/50 (CLS + MIL)' in report
    assert 'MIL (Patch Head only)' in report
    assert 'CLS only' in report


def test_format_summary_table_multi_dataset_ablation_leaderboard():
    res1 = {
        'video_auc': 0.90, 'auc': 0.89, 'video_eer': 0.12, 'eer': 0.13, 'acc': 0.85,
        'bdg_f_mil_video_auc': 0.96, 'bdg_f_mil_auc': 0.95, 'bdg_f_mil_video_eer': 0.06, 'bdg_f_mil_eer': 0.07, 'bdg_f_mil_video_acc': 0.91, 'bdg_f_mil_video_ap': 0.94,
        'bdg_cls_mil_video_auc': 0.94, 'bdg_cls_mil_auc': 0.93, 'bdg_cls_mil_video_eer': 0.08, 'bdg_cls_mil_eer': 0.09, 'bdg_cls_mil_video_acc': 0.89, 'bdg_cls_mil_video_ap': 0.92,
        'ens_cls_mil_video_auc': 0.92, 'ens_cls_mil_auc': 0.91, 'ens_cls_mil_video_eer': 0.10, 'ens_cls_mil_eer': 0.11, 'ens_cls_mil_video_acc': 0.87, 'ens_cls_mil_video_ap': 0.90,
        'ens_f_mil_video_auc': 0.93, 'ens_f_mil_auc': 0.92, 'ens_f_mil_video_eer': 0.09, 'ens_f_mil_eer': 0.10, 'ens_f_mil_video_acc': 0.88, 'ens_f_mil_video_ap': 0.91,
        'feature_fusion_video_auc': 0.90, 'feature_fusion_auc': 0.89, 'feature_fusion_video_eer': 0.12, 'feature_fusion_eer': 0.13, 'feature_fusion_video_acc': 0.85, 'feature_fusion_video_ap': 0.88,
        'mil_video_auc': 0.88, 'mil_auc': 0.87, 'mil_video_eer': 0.14, 'mil_eer': 0.15, 'mil_video_acc': 0.83, 'mil_video_ap': 0.86,
        'cls_only_video_auc': 0.86, 'cls_only_auc': 0.85, 'cls_only_video_eer': 0.16, 'cls_only_eer': 0.17, 'cls_only_video_acc': 0.81, 'cls_only_video_ap': 0.84,
    }
    res2 = {k: (v - 0.04 if isinstance(v, float) else v) for k, v in res1.items()}
    dataset_results = [('Celeb-DF-v2', res1), ('FaceShifter', res2)]

    table_str = format_summary_table(dataset_results, 'bias_sspanet_feat_mil', 1024)
    # Must have primary dataset summary table
    assert 'Dataset' in table_str
    assert 'Celeb-DF-v2' in table_str
    assert 'FaceShifter' in table_str
    assert 'AVERAGE' in table_str

    # Must have Ablation Leaderboard Summary across datasets
    assert '[Ablation Leaderboard Summary - Average Across 2 Datasets (Ranked by Video AUC)]:' in table_str
    assert 'Adaptive BDG (F + MIL)' in table_str
    assert 'Adaptive BDG (CLS + MIL)' in table_str
    assert 'Ensemble 50/50 (F + MIL)' in table_str
    assert 'Ensemble 50/50 (CLS + MIL)' in table_str
    assert 'Feature Fusion (F)' in table_str
    assert 'MIL (Patch Head only)' in table_str
    assert 'CLS only' in table_str
    # Rank 1 must be BDG (F + MIL) with average video auc = (0.96 + 0.92)/2 = 94.00%
    assert '94.00%' in table_str

