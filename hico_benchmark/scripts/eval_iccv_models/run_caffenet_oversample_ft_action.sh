#!/bin/bash

mkdir -p logs/eval_iccv_models

LOG="logs/eval_iccv_models/caffenet_oversample_ft_action.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time matlab -r "feat_type = 'feat_iccv_models-caffenet_oversample_ft_action'; train_svm_vo; eval_default_run; eval_ko_run;"
