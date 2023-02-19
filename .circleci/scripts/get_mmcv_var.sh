#!/bin/bash

TORCH=$1
CUDA=$2

# 10.2 -> cu102
MMCV_CUDA="cu`echo ${CUDA} | tr -d '.'`"

# MMCV only provides pre-compiled packages for torch 1.x.0
# which works for any subversions of torch 1.x.
# We force the torch version to be 1.x.0 to ease package searching
# and avoid unnecessary rebuild during MMCV's installation.
TORCH_VER_ARR=(${TORCH//./ })
TORCH_VER_ARR[2]=0
printf -v MMCV_TORCH "%s." "${TORCH_VER_ARR[@]}"
MMCV_TORCH=${MMCV_TORCH%?}  # Remove the last dot

echo "export MMCV_CUDA=${MMCV_CUDA}" >> $BASH_ENV
echo "export MMCV_TORCH=${MMCV_TORCH}" >> $BASH_ENV
