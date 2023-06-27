#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

# launch core setting
KERNEL_NUM=$(($(nproc)/GPUS))

# dist env setting
export WORLD_SIZE=$((NNODES*GPUS))
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$PORT
LOCAL_RANK_START=$((NODE_RANK*GPUS))
LOCAL_RANK_END=$((LOCAL_RANK_START+GPUS))

for((RANK_ID=LOCAL_RANK_START;RANK_ID<LOCAL_RANK_END;RANK_ID++))
do
	export RANK=$RANK_ID
	PID_START=$((KERNEL_NUM*(RANK_ID%GPUS)))
	PID_END=$((PID_START+KERNEL_NUM-1))
	nohup taskset -c $PID_START-$PID_END \
	python $(dirname "$0")/train.py \
	$CONFIG \
	--seed 0 \
	--launcher pytorch ${@:3} \
	&
done
