#!/bin/bash

mkdir -p logs/eval_iccv_models

LOG="logs/eval_iccv_models/caffenet_full_imagenet.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time matlab -r "feat_type = 'feat_iccv_models-caffenet_full_imagenet'; train_svm_vo; eval_default_run; eval_ko_run;"
