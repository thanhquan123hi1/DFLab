"""Run BGATE experiments independently of baseline train.py."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=None, help='BGATE YAML; required for a new run')
    parser.add_argument('--output-root', default='runs', help='A bgate_v1 namespace is always appended')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--rgb-dir')
    parser.add_argument('--dataset-json-folder')
    parser.add_argument('--batch-size',type=int)
    parser.add_argument('--eval-batch-size',type=int)
    parser.add_argument('--workers',type=int)
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--device',default=None,help='cuda or cpu')
    parser.add_argument('--deterministic',action='store_true')
    parser.add_argument('--resume',help='Resume only an epoch-boundary BGATE last.pt in its original run')
    args = parser.parse_args()
    if int(os.environ.get('WORLD_SIZE','1')) > 1:
        parser.error('BGATE v1 is single-device; do not launch with torchrun/DDP')
    if args.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from detectors.bgate_mil_detector import MODELS,ARCHITECTURE
    from bgate_common import training_config,seed_everything,make_loader,manifest,check_overlap,ROOT
    from bgate_logging import new_run,logger_for,metadata,load_checkpoint,source_hashes,verify_resume
    from trainer.bgate_trainer import BGateTrainer
    from metrics.utils import write_json

    checkpoint = None
    if args.resume:
        if args.config or args.seed is not None or args.deterministic or any(v is not None for v in
            (args.rgb_dir,args.dataset_json_folder,args.batch_size,args.eval_batch_size,args.workers,args.epochs)):
            parser.error('Resume uses saved config; no training overrides are allowed')
        path = Path(args.resume).resolve()
        checkpoint = load_checkpoint(path)
        run = path.parent.parent
        if path.name != 'last.pt' or path.parent.name != 'checkpoints' or not (run/'run.json').is_file():
            parser.error('Resume requires checkpoints/last.pt from a BGATE run')
        if run != Path(checkpoint['run_dir']).resolve():
            parser.error('Resume must target the original run directory')
        cfg = checkpoint['config']
        if cfg.get('deterministic'):
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    else:
        if not args.config:
            parser.error('--config is required for a new run')
        cfg = training_config(args.config)
        for key,value in dict(manualSeed=args.seed,rgb_dir=args.rgb_dir,
            dataset_json_folder=args.dataset_json_folder,train_batchSize=args.batch_size,
            test_batchSize=args.eval_batch_size,workers=args.workers,nEpochs=args.epochs).items():
            if value is not None:
                cfg[key] = value
        cfg['deterministic'] = args.deterministic
        if cfg['nEpochs']<1 or cfg['train_batchSize']<1 or cfg['test_batchSize']<1 or cfg['workers']<0:
            parser.error('Epochs/batch sizes must be positive and workers nonnegative')
        if cfg['model_name'] not in MODELS:
            parser.error('Only BGATE model names are accepted')
        run = new_run(args.output_root,cfg['model_name'],cfg['manualSeed'])
    if cfg['model_name'] not in MODELS:
        parser.error('Only BGATE model names are accepted')
    device = torch.device(args.device or ('cuda' if cfg.get('cuda',True) and torch.cuda.is_available() else 'cpu'))
    # Prevent two trainers appending to the same run, including concurrent resume.
    lock = run/'.training.lock'
    with lock.open('x',encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    trainer = None
    try:
        logger = logger_for(run/'train/training.log')
        seed_everything(cfg['manualSeed'],cfg.get('deterministic',False))
        train_loader = make_loader(cfg,'train')
        val_loader = make_loader(cfg,'test','FaceForensics++','val')
        check_overlap(train_loader,val_loader)
        manifests = dict(train=manifest(train_loader),validation=manifest(val_loader))
        if checkpoint:
            import json
            original = json.loads((run/'run.json').read_text(encoding='utf-8'))
            verify_resume(original,cfg,manifests,source_hashes(ROOT))
        else:
            write_json(run/'run.json',dict(architecture=ARCHITECTURE,config=cfg,
                train_seed=cfg['manualSeed'],manifests=manifests,device=str(device),
                source_hashes=source_hashes(ROOT),selection='FF++ val / bounded_gate video_auc',
                **metadata(ROOT.parent)))
        model = MODELS[cfg['model_name']](cfg)
        trainer = BGateTrainer(model,cfg,run,device,logger)
        if checkpoint:
            trainer.resume(checkpoint)
            write_json(run/'train/resume_events.jsonl',dict(
                checkpoint=str(path),next_epoch=trainer.start_epoch,global_step=trainer.step,
                **metadata(ROOT.parent)),append=True)
        write_json(run/'trainable_parameters.json',model.trainable_counts)
        logger.info('Run: %s; model=%s seed=%s trainable=%s',run,cfg['model_name'],cfg['manualSeed'],model.trainable_counts)
        trainer.fit(train_loader,val_loader)
    finally:
        if trainer:
            trainer.close()
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
