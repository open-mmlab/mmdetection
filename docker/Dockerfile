FROM nvidia/cuda:10.1-devel-ubuntu18.04

# Set up locale to prevent bugs with encoding
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update && \
      apt-get install -y apt-utils && \
      apt-get install -y \
            wget \
            curl \
            libsm6 \
            libxext6 \
            libxrender-dev \
            python3 \
            python3-pip && \
      rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY ./requirements.txt /workspace/requirements.txt
RUN pip3 install -r /workspace/requirements.txt

COPY . /workspace/

RUN cd /workspace/ && \
    PYTHON=python3 bash ./compile.sh && \
    pip3 install -e . && \
    cd ..

RUN mkdir /.cache && chmod -R a+rw /.cache/
WORKDIR /workspace