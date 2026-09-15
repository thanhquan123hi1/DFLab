python training/train.py \
--detector_path ./training/config/detector/ln_sspanet_mil.yaml  \
--train_dataset "FaceForensics++" \
--test_dataset  "Celeb-DF-v2" \
"$@"
