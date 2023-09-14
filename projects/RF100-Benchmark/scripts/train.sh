#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
WORK_DIRS=${3:-'work_dirs'}
RETRY_PATH=${RETRY_PATH:-''}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

datasets=$(pwd)/rf100
export PYTHONPATH="../..":$PYTHONPATH

DEBUG=0
# example
datasets_list=("acl-x-ray", "tweeter-profile")

if [ -n "$RETRY_PATH" ]; then
    DEBUG=1
    datasets_list=()
    while IFS= read -r line; do
      if [ -n "$line" ]; then
          datasets_list+=("$line")
      fi
    done < "$RETRY_PATH"
fi

if [ "$DEBUG" == 1 ]; then
  echo "current training dataset list is: ${datasets_list[@]}"
else
  echo "Currently training with the full dataset."
fi
echo "=============================================="

for dataset in $(ls $datasets)
  do
    # You can customize string_list to train only specific datasets.
    if [ "$DEBUG" == 1 ]; then
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
    echo "=============================================="
  done

#rm -rf temp_configs
echo "Done training all the datasets"
