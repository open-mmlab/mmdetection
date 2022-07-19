ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# redeclare arguments are lost after FROM build stage
ARG PYTORCH
ARG CUDNN
ARG CUDA
ARG MMCV="1.5.2"

RUN echo "building pytorch=$PYTORCH cuda=$CUDA cudnn=$CUDNN mmcv=$MMCV"
RUN echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> /etc/bash.bashrc

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# FIXME update will fail due to invalid signature, 
# ingore for now with ';' instead of '&&' could be solved later like this
#RUN apt update ; apt install wget -y \
    #&& apt-key del 7fa2af80 \
    #&& wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \
    #&& dpkg -i cuda-keyring_1.0-1_all.deb \
    #&& rm -rf /var/lib/apt/lists/*
RUN apt-get update ; \
    apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
ENV FORCE_CUDA="1"
ENV MMCV_WITH_OPS="1"
RUN pip install --no-cache-dir --upgrade pip wheel setuptools
RUN pip install --no-cache-dir mmcv-full==${MMCV} -f https://download.openmmlab.com/mmcv/dist/cu$(echo ${CUDA} | sed 's/\.//')/torch${PYTORCH}/index.html

# Install MMDetection
RUN conda clean --all
RUN git clone https://github.com/logivations/mmdetection.git /mmdetection
WORKDIR /mmdetection
RUN pip install --no-cache-dir -r requirements/build.txt
RUN pip install --no-cache-dir -e .
RUN python3 mmdet/utils/collect_env.py
RUN pip install black

RUN mkdir /data /code
WORKDIR /code
