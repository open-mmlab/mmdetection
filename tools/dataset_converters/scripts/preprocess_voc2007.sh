#!/usr/bin/env bash

DOWNLOAD_DIR=$1
DATA_ROOT=$2

tar -xvf $DOWNLOAD_DIR/OpenDataLab___PASCAL_VOC2007/raw/VOCtrainval_06-Nov-2007.tar -C $DATA_ROOT
tar -xvf $DOWNLOAD_DIR/OpenDataLab___PASCAL_VOC2007/raw/VOCtestnoimgs_06-Nov-2007.tar -C $DATA_ROOT
rm -rf $DOWNLOAD_DIR/OpenDataLab___PASCAL_VOC2007
