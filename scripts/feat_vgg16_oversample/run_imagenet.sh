#!/bin/bash

export PYTHONPATH=./caffe/python

mkdir -p logs/feat_vgg16_oversample

LOG="logs/feat_vgg16_oversample/imagenet.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ./lib/extract_batch_hico.py \
  ./external/hico_20150920/images \
  ./output/feat_vgg16_oversample/imagenet \
  1 \
  1 \
  --gpu \
  --model_def ./models/VGG16/deploy.prototxt \
  --pretrained_model ./data/imagenet_models/VGG16.v2.caffemodel \
  --mean_file setmean-VGG16
