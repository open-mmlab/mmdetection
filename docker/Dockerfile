FROM nvidia/cuda:10.0-devel-ubuntu18.04 AS build
ARG PYTHON_VERSION=3.7
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda100 && \
     /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH

RUN conda install -y -c pytorch pytorch torchvision && conda clean -ya
RUN conda install glib -y && conda clean -ya

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd -g ${GROUP_ID} user && useradd -m -u ${USER_ID} -g user user

ENV LD_LIBRARY_PATH ''
ENV LIBRARY_PATH ''

RUN echo /opt/conda/lib/ > /etc/ld.so.conf.d/conda && ldconfig

COPY --chown=user:user ./ /home/user/mmdetection

USER user

RUN pip --no-cache-dir install --user -e /home/user/mmdetection && \
        pip uninstall -y opencv-python && \
        pip --no-cache-dir install --user opencv-python-headless && \
        rm -rf /home/user/mmdetection/build


FROM ubuntu:18.04

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.0"

# nvidia-driver-installer uses /usr/local/nvidia/ directory to install/mount nvidia binaries and libraries
ENV PATH /home/user/.local/bin:/opt/conda/bin:/usr/local/nvidia/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd -g ${GROUP_ID} user && useradd -m -u ${USER_ID} -g user user

RUN apt-get update && apt-get install -y --no-install-recommends \
         ca-certificates \
         git && \
     rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/conda /opt/conda

RUN echo /opt/conda/lib/ > /etc/ld.so.conf.d/conda && \
    echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf && ldconfig

COPY --from=build --chown=user:user /home/user /home/user

USER user
