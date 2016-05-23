#!/bin/bash

mkdir -p logs/eval_vgg16_center

LOG="logs/eval_vgg16_center/ft_object_single.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time matlab -r "feat_type = 'feat_vgg16_center-ft_object_single'; train_svm_vo; eval_default_run; eval_ko_run;"
