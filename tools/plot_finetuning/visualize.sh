#!/bin/bash

base_dir="$(dirname $0)/../../"
log_file=$1;
path="${log_file%/*}"
name="${log_file##*/}"
# name="${name%.*}"
# echo $name
# echo $path
echo "Saving plots to ${path}/${name}_plot/ ..."
python ${base_dir}/tools/plot_finetuning/plot_training_log.py -2 ${path}/${name}_plot/ ${log_file}
