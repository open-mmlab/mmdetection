ARG PYTORCH="1.3"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 cmake libboost-python-dev\
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install mmdetection
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
RUN conda config --set show_channel_urls yes
RUN conda install cython  -y && conda clean --all
RUN pip install Flask==1.1.1 dlib py_eureka_client==0.7.4 -i https://mirrors.aliyun.com/pypi/simple/ 
#RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection

WORKDIR /mmdetection
COPY mmdetection .
RUN pip install --no-cache-dir -e .

#COPY data/student/*.py   data/student/
#COPY work_dirs/cascade_rcnn_x101_64x4d_fpn_1x/latest.pth  work_dirs/cascade_rcnn_x101_64x4d_fpn_1x/
#COPY eureka_rest.py .


CMD ["python", "eureka_rest.py"]

# run
# docker run --name mmdet2_consloe  --gpus all -it -p 7100:7100  -v /home/AI/mmdet2/:/mmdetection/app  mmdet2:v1.0 /bin/bash
# docker run --name mmdet2_consloe  --gpus all -it -p 7100:7100  -v /home/mmde2:/mmdetection/app  mmdet2:v1.0 /bin/bash
# docker run --name mmdet2_app  --gpus all -it -p 7100:7100  -v /home/mmde2:/mmdetection/app  mmdet2:v1.0
# docker run --name mmdet2_app      --gpus all -it -d -p 7100:7100  -v  /home/AI/mmdet2/eureka_rest.py:/mmdetection/eureka_rest.py   -v  /home/AI/mmdet2/work_dirs:/mmdetection/work_dirs -v  /home/AI/mmdet2/data:/mmdetection/data mmdet2:v1.0

# build 
# docker build -t mmdet2:v1.0 .
# docker build --no-cache -t mmdet2:v1.0 .
