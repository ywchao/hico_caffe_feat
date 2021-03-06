#!/bin/bash

export PYTHONPATH=./caffe/python

mkdir -p logs/feat_vgg16_full

LOG="logs/feat_vgg16_full/ft_action_single.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ./lib/extract_batch_hico.py \
  ./external/hico_20150920/images \
  ./output/feat_vgg16_full/ft_action_single \
  1 \
  1 \
  --gpu \
  --model_def ./models/VGG16/deploy.prototxt \
  --pretrained_model ./output/ft_vgg16/ft_action_single/vgg16_iter_50000.caffemodel \
  --mean_file setmean-VGG16 \
  --crop_mode full
