# syntax=docker/dockerfile:1.0.0-experimental

FROM acesdev/algo-base:awscli2.1.25-poetry1.1.13-tensorrt8.2.3.0-python3.8.7-cuda11.1-cudnn8-devel-ubuntu18.04

# This is for cuda 11.1. In poetry.lock, set for cuda10.2 (setting of project.toml).
RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install tqdm

# Install MMCV
RUN pip install --no-cache-dir --upgrade pip wheel setuptools
RUN pip install --no-cache-dir mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
