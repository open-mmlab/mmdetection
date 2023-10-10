#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
WORK_DIRS=${3:-'work_dirs'}
RETRY_PATH=${RETRY_PATH:-''}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PARTITION=${PARTITION:-'mm_dev'}
JOB_NAME=${JOB_NAME:-'benchmark'}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}

datasets=$(pwd)/rf100
export PYTHONPATH="../..":$PYTHONPATH

DEBUG=0
# example
datasets_list=('bacteria-ptywi', 'circuit-elements', 'marbles', 'printed-circuit-board', 'solar-panels-taxvb')

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

    srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u  ../../tools/train.py "temp_configs/$dataset.py"  --work-dir="$WORK_DIRS/$dataset" --launcher="slurm" ${@:4}

    echo "=============================================="
  done

#rm -rf temp_configs
echo "Done training all the datasets"
