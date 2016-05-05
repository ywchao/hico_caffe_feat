#!/bin/bash

export PYTHONPATH=./caffe/python

mkdir -p logs/feat_iccv_models

LOG="logs/feat_iccv_models/caffenet_oversample_ft_verb.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ./lib/extract_batch_hico.py \
  ./external/hico_20150920/images \
  ./output/feat_iccv_models/caffenet_oversample_ft_verb \
  1 \
  1 \
  --gpu \
  --pretrained_model ./data/iccv_models/caffenet_ft_verb_single_iter_50000.caffemodel
