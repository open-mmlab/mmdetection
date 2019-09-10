ARG PYTORCH="1.1.0"
ARG CUDA="10.0"
ARG CUDNN="7.5"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 \
 tmux \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install mmdetection
RUN conda install cython -y && conda clean --all \
    && pip install --ignore-installed --no-cache-dir --upgrade torch>=1.1\
    torchvision \
    && pip install --ignore-installed --no-cache-dir mmcv>=0.2.10 \
    numpy \
    matplotlib \
    six \
    terminaltables \
    pycocotools \
    imagecorruptions \
    gpustat \
    seaborn \
    tensorboard
ENV TERM=xterm-256color
ENV PYTHONPATH=$PYTHONPATH:/workspace/mmdetection
ENV OMP_NUM_THREADS=4
EXPOSE 22
EXPOSE 80