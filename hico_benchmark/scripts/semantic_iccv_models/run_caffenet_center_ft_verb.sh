#!/bin/bash

mkdir -p logs/semantic_iccv_models

LOG="logs/semantic_iccv_models/caffenet_center_ft_verb.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time matlab -r "label_type = 'sep_vb'; feat_type = 'feat_iccv_models-caffenet_center_ft_verb'; train_svm_sep;"
time matlab -r "label_type = 'sep_nn'; feat_type = 'feat_iccv_models-caffenet_center_ft_verb'; train_svm_sep;"
time matlab -r "feat_type = 'feat_iccv_models-caffenet_center_ft_verb'; semantic_cache_score;"
time matlab -r "feat_type = 'feat_iccv_models-caffenet_center_ft_verb'; use_parfor = false; semantic_default_run; semantic_ko_run;"
