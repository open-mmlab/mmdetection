#!/usr/bin/env bash

DOWNLOAD_DIR=$1
DATA_ROOT=$2

tar -xvf $DOWNLOAD_DIR/OpenDataLab___PASCAL_VOC2012/raw/VOCtrainval_11-May-2012.tar -C $DATA_ROOT
tar -xvf $DOWNLOAD_DIR/OpenDataLab___PASCAL_VOC2012/raw/VOC2012test.tar -C $DATA_ROOT
rm -rf $DOWNLOAD_DIR/OpenDataLab___PASCAL_VOC2012
