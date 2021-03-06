#!/bin/bash

export PYTHONPATH=./caffe/python

mkdir -p logs/feat_iccv_models

LOG="logs/feat_iccv_models/caffenet_full_ft_action.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ./lib/extract_batch_hico.py \
  ./external/hico_20150920/images \
  ./output/feat_iccv_models/caffenet_full_ft_action \
  1 \
  1 \
  --gpu \
  --pretrained_model ./data/iccv_models/caffenet_ft_action_single_iter_50000.caffemodel \
  --crop_mode full
