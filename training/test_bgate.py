"""Evaluate five BGATE outputs into a fresh, isolated evaluation directory."""
import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',required=True)
    parser.add_argument('--datasets',nargs='+',default=['Celeb-DF-v2','DFDC'])
    parser.add_argument('--output-root',default=None)
    parser.add_argument('--rgb-dir')
    parser.add_argument('--dataset-json-folder')
    parser.add_argument('--batch-size',type=int)
    parser.add_argument('--workers',type=int)
    parser.add_argument('--device',default=None)
    parser.add_argument('--evaluation-seed',type=int,default=1024)
    parser.add_argument('--max-samples',type=int,help='Smoke test only; marked partial, not a reportable full AUC')
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples<1:
        parser.error('--max-samples must be positive')
    if len(set(args.datasets)) != len(args.datasets) or any('/' in x or '\\' in x or x in ('.','..') for x in args.datasets):
        parser.error('Dataset names must be unique safe names')
    import torch
    import numpy as np
    from detectors.bgate_mil_detector import MODELS,ARCHITECTURE
    from bgate_common import make_loader,seed_everything,manifest
    from bgate_logging import load_checkpoint,unique_dir,logger_for,sha256
    from metrics.bgate_evaluation import evaluate
    from metrics.utils import write_json
    path = Path(args.weights).resolve()
    checkpoint = load_checkpoint(path)
    cfg = dict(checkpoint['config'])
    for key,value in dict(rgb_dir=args.rgb_dir,dataset_json_folder=args.dataset_json_folder,
                          test_batchSize=args.batch_size,workers=args.workers).items():
        if value is not None:
            cfg[key] = value
    if cfg['test_batchSize']<1 or cfg['workers']<0:
        parser.error('Invalid batch size/workers')
    if cfg.get('deterministic'):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    seed_everything(args.evaluation_seed,cfg.get('deterministic',False))
    device = torch.device(args.device or ('cuda' if cfg.get('cuda',True) and torch.cuda.is_available() else 'cpu'))
    model = MODELS[cfg['model_name']](cfg).to(device)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    model.set_epoch(checkpoint['epoch'])
    if args.output_root:
        parent = Path(args.output_root)/'bgate_v1'/cfg['model_name']/f"seed_{checkpoint['train_seed']}"/'tests'
    else:
        run = path.parent.parent
        if path.parent.name!='checkpoints' or not (run/'run.json').exists():
            parser.error('Use --output-root for a checkpoint outside its BGATE run')
        parent = run/'tests'
    output = unique_dir(parent)
    logger = logger_for(output/'testing.log')
    info = dict(architecture=ARCHITECTURE,config=cfg,checkpoint=str(path),
        checkpoint_sha256=sha256(path),train_seed=checkpoint['train_seed'],
        evaluation_seed=args.evaluation_seed,epoch=checkpoint['epoch'],
        gate_active=model.gate_active,primary_output='bounded_gate',fixed_ensemble_weight=.5,
        fixed_ensemble_base='feature_fusion',max_samples=args.max_samples,manifests={})
    write_json(output/'evaluation.json',info)
    rows = []
    for name in args.datasets:
        loader = make_loader(cfg,'test',name,'test')
        info['manifests'][name] = manifest(loader)
        metrics,arrays = evaluate(model,loader,device,args.max_samples)
        write_json(output/name/'metrics.json',metrics)
        np.savez_compressed(output/name/'predictions.npz',**arrays)
        for branch,m in metrics['branches'].items():
            rows.append(dict(dataset=name,branch=branch,video_auc=m['video_auc'],video_eer=m['video_eer'],
                frame_auc=m['auc'],video_n=m['video_n'],frame_n=m['n'],partial=metrics['partial']))
        logger.info('%s video AUC: %s',name,{k:round(v['video_auc'],6) for k,v in metrics['branches'].items()})
        write_json(output/'evaluation.json',info)
        with (output/'summary.csv').open('w',newline='',encoding='utf-8') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    logger.info('Evaluation saved to %s',output)


if __name__ == '__main__':
    main()
