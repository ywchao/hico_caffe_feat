#!/bin/bash

mkdir -p logs/semantic_caffenet_center

LOG="logs/semantic_caffenet_center/imagenet.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time matlab -r "label_type = 'sep_vb'; feat_type = 'feat_caffenet_center-imagenet'; train_svm_sep;"
time matlab -r "label_type = 'sep_nn'; feat_type = 'feat_caffenet_center-imagenet'; train_svm_sep;"
time matlab -r "feat_type = 'feat_caffenet_center-imagenet'; semantic_cache_score;"
time matlab -r "feat_type = 'feat_caffenet_center-imagenet'; use_parfor = true; semantic_default_run; semantic_ko_run;"
