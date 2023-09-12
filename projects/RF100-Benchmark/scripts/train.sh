#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
WORK_DIRS=${3:-'work_dirs'}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

DEBUG=0

datasets=$(pwd)/rf100
export PYTHONPATH="../..":$PYTHONPATH

for dataset in $(ls $datasets)
  do
    # You can customize string_list to train only specific datasets.
    if [ "$DEBUG" == 1 ]; then
      # example
      datasets_list=("acl-x-ray", "tweeter-profile")
      if [[ ! " ${datasets_list[@]} " =~ "$dataset" ]]; then
        continue
      fi
    fi

    echo "Training on $dataset"
    python $(pwd)/scripts/create_new_config.py $CONFIG $dataset
    if [ "$GPUS" == 1 ]; then
       python ../../tools/train.py "temp_configs/$dataset.py" --work-dir "$WORK_DIRS/$dataset" ${@:4}
    else
        python -m torch.distributed.launch \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --nproc_per_node=$GPUS \
            --master_port=$PORT \
            ../../tools/train.py \
            "temp_configs/$dataset.py" \
            --launcher pytorch --work-dir "$WORK_DIRS/$dataset" ${@:4}
    fi
  done

rm -rf temp_configs
echo "Done training all the datasets"
