ARG PYTORCH="1.4"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install mmdetection
RUN conda clean --all
RUN git clone --recursive --branch ote https://github.com/opencv/mmdetection.git /mmdetection
WORKDIR /mmdetection
ENV FORCE_CUDA="1"
RUN pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
RUN pip install --ignore-installed --no-cache-dir -e .['all']

# Install OpenVINO
RUN apt update && \
    apt install -y \
        libgtk-3-dev \
        cpio \
        unzip \
        wget \
        curl

ARG OPENVINO_LINK=http://registrationcenter-download.intel.com/akdlm/irc_nas/16670/l_openvino_toolkit_p_2020.3.194.tgz
RUN mkdir -p /tmp/openvino_installer && cd /tmp/openvino_installer && \
    wget --no-verbose $OPENVINO_LINK && \
    tar xf l_openvino_toolkit*.tgz && \
    rm l_openvino_toolkit*.tgz && \
    cd l_openvino_toolkit* && \
    bash ./install.sh -s silent.cfg --accept_eula && \
    cd /opt/intel/openvino/install_dependencies/ && \
    dpkg -i intel-*.deb && \
    ldconfig && \
    rm -rf /tmp/openvino_installer && \
    pip install -r /opt/intel/openvino/deployment_tools/model_optimizer/requirements_onnx.txt && \
    echo "source /opt/intel/openvino/bin/setupvars.sh" >> /etc/profile

# Prepare workspace to use for any user
ENV BASH_ENV "/etc/profile"
ENV USER docker
RUN chmod -R 777 /workspace
ENV HOME /workspace
WORKDIR /workspace
