#!/bin/bash

export PYTHONPATH=./caffe/python

mkdir -p logs/feat_iccv_models

LOG="logs/feat_iccv_models/caffenet_center_ft_object.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ./lib/extract_batch_hico.py \
  ./external/hico_20150920/images \
  ./output/feat_iccv_models/caffenet_center_ft_object \
  1 \
  1 \
  --gpu \
  --pretrained_model ./data/iccv_models/caffenet_ft_object_single_iter_50000.caffemodel \
  --crop_mode center
