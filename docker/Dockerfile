ARG PYTORCH="1.1.0"
ARG CUDA="10.0"
ARG CUDNN="7.5"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# ENV CUDA_HOME "/usr/local/cuda"

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6

# Install mmdetection
RUN conda install cython -y
# RUN pip install pycocotools mmcv
RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
WORKDIR /mmdetection
RUN pip install -e .
