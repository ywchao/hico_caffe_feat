#!/bin/bash

export PYTHONPATH=./caffe/python

mkdir -p logs/feat_caffenet_oversample

LOG="logs/feat_caffenet_oversample/ft_action_single.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ./lib/extract_batch_hico.py \
  ./external/hico_20150920/images \
  ./output/feat_caffenet_oversample/ft_action_single \
  1 \
  1 \
  --gpu \
  --pretrained_model ./output/ft_caffenet/ft_action_single/caffenet_iter_50000.caffemodel
