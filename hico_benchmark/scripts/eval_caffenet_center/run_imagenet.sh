#!/bin/bash

mkdir -p logs/eval_caffenet_center

LOG="logs/eval_caffenet_center/imagenet.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time matlab -r "feat_type = 'feat_caffenet_center-imagenet'; train_svm_vo; eval_default_run; eval_ko_run;"
