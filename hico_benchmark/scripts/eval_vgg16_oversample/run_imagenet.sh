#!/bin/bash

mkdir -p logs/eval_vgg16_oversample

LOG="logs/eval_vgg16_oversample/imagenet.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time matlab -r "feat_type = 'feat_vgg16_oversample-imagenet'; train_svm_vo; eval_default_run; eval_ko_run;"
