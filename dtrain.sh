python3 -m torch.distributed.launch --nproc_per_node=2 training/train.py \
--detector_path ./training/config/detector/ln_sspanet_mil.yaml \
--train_dataset FaceForensics++ \
--test_dataset Celeb-DF-v2 \
--ddp \
"$@"

