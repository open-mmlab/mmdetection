#!/bin/bash
####
# workon bepro-video-util (GCP virtualenv)

# below will be added to above venv
pip install torch==1.3.1 torchvision==0.4.2
pip install cython
pip install pycocotools==2.0.0
pip install mmcv-full

mkdir data

/# /snap/bin/gsutil
wget https://storage.googleapis.com/bepro-dev/scouting_feed/mmdetection/libnccl-dev_2.4.7-1%2Bcuda10.0_amd64.deb
wget https://storage.googleapis.com/bepro-dev/scouting_feed/mmdetection/libnccl2_2.4.7-1%2Bcuda10.0_amd64.deb
sudo dpkg -i libnccl2_2.4.7-1+cuda10.0_amd64.deb
sudo dpkg -i libnccl-dev_2.4.7-1+cuda10.0_amd64.deb
sudo apt update
sudo apt -y install libnccl2=2.4.7-1+cuda10.0 libnccl-dev=2.4.7-1+cuda10.0

mkdir demo/dump

## error fix 
# nvcc fatal : Unsupported gpu architecture 'compute_75'
# export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}^C
# export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
python setup.py develop

cp -r configs ~/bepro-video-util 
cp crcnn_r50_bepro_stitch.pth ~/bepro-video-util

## test with
# python demo/mmdetection_demo.py $input_vid configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_bepro.py crcnn_r50_bepro_stitch.pth 0 1000 ./tmp/
