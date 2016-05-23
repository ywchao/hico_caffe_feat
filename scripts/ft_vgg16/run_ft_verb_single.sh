#!/bin/bash

if [ $# -eq 0 ]; then
  gpu_id=0
else
  gpu_id=$1
fi

mkdir -p logs/ft_vgg16

LOG="logs/ft_vgg16/ft_verb_single.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# make directory
mkdir -p output/ft_vgg16/ft_verb_single

# prepare lmdb
time matlab -r "ft_prepare_lmdb('verb', 'single'); exit;"

# finetune
time ./caffe/build/tools/caffe train \
  -solver models/VGG16/solver_ft_verb_single.prototxt \
  -weights data/imagenet_models/VGG16.v2.caffemodel \
  -gpu $gpu_id
