#!/bin/bash

if [ $# -eq 0 ]; then
  gpu_id=0
else
  gpu_id=$1
fi

mkdir -p logs/ft_caffenet

LOG="logs/ft_caffenet/ft_action_single.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# make directory
mkdir -p output/ft_caffenet/ft_action_single

# prepare lmdb
time matlab -r "ft_prepare_lmdb('action', 'single'); exit;"

# finetune
time ./caffe/build/tools/caffe train \
  -solver models/bvlc_reference_caffenet/solver_ft_action_single.prototxt \
  -weights caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
  -gpu $gpu_id
