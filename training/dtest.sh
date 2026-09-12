# Chạy trên 2 GPU
torchrun --nproc_per_node=2 training/test.py \
  --ddp \
  --detector_path training/config/detector/biasln.yaml \
  --test_dataset Celeb-DF-v2\
  --weights_path /path/to/biasln_weights.pth\
  --save_feat


