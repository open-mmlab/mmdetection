#!/bin/bash
__heredoc__="""

This script builds mmdetection manylinux2014 wheels.


By default this script builds wheels for the current python version.
To build wheels for a specific python version run:

    MB_PYTHON_TAG=cp38-cp38 ./run_manylinux_build.sh
    MB_PYTHON_TAG=cp37-cp37m ./run_manylinux_build.sh
    MB_PYTHON_TAG=cp36-cp36m ./run_manylinux_build.sh
    MB_PYTHON_TAG=cp35-cp35m ./run_manylinux_build.sh
    MB_PYTHON_TAG=cp27-cp27m ./run_manylinux_build.sh


notes:

    Manylinux repo: https://github.com/pypa/manylinux 

    Win + Osx repo: https://github.com/mavlink/MAVSDK-Python not sure how to do this yet

    docker pull quay.io/pypa/manylinux2010_x86_64:latest

    docker pull pytorch/manylinux-cuda101

    references on cuda manylinux images:
        https://hub.docker.com/u/soumith/

        Install cuda in base manylinux

        wget https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-repo-rhel7-10.0.130-1.x86_64.rpm

        https://github.com/pytorch/builder/blob/master/manywheel/Dockerfile_101 

        https://discuss.python.org/t/the-next-manylinux-specification/1043
"""


#DOCKER_IMAGE=${DOCKER_IMAGE:="quay.io/pypa/manylinux2010_x86_64:latest"}
#DOCKER_IMAGE=${DOCKER_IMAGE:="soumith/manylinux-cuda101:latest"}
DOCKER_IMAGE=${DOCKER_IMAGE:="pytorch/manylinux-cuda101"}

# Valid multibuild python versions are:
# cp27-cp27m  cp27-cp27mu  cp34-cp34m  cp35-cp35m  cp36-cp36m  cp37-cp37m, cp38-cp38m
MB_PYTHON_TAG=${MB_PYTHON_TAG:=$(python -c "import setup; print(setup.native_mb_python_tag())")}
NAME=${NAME:=$(python -c "import setup; print(setup.NAME)")}
VERSION=${VERSION:=$(python -c "import setup; print(setup.VERSION)")}
echo "
MB_PYTHON_TAG = $MB_PYTHON_TAG
DOCKER_IMAGE = $DOCKER_IMAGE
VERSION = $VERSION
NAME = $NAME
"

if [ "$_INSIDE_DOCKER" != "YES" ]; then

    set -e
    docker run --runtime=nvidia --rm \
        -v $PWD:/io \
        -e _INSIDE_DOCKER="YES" \
        -e MB_PYTHON_TAG="$MB_PYTHON_TAG" \
        -e NAME="$NAME" \
        -e VERSION="$VERSION" \
        $DOCKER_IMAGE bash -c 'cd /io && ./run_manylinux_build.sh'

    __interactive__='''
    # notes for running / debugging interactively 

    docker run --runtime=nvidia --rm \
        -v $PWD:/io \
        -e _INSIDE_DOCKER="YES" \
        -e MB_PYTHON_TAG="$MB_PYTHON_TAG" \
        -e NAME="$NAME" \
        -e VERSION="$VERSION" \
        -it $DOCKER_IMAGE bash

    set +e
    set +x
    '''

    BDIST_WHEEL_PATH=$(ls wheelhouse/$NAME-$VERSION-$MB_PYTHON_TAG*.whl)
    echo "BDIST_WHEEL_PATH = $BDIST_WHEEL_PATH"
else
    #set -x
    #set -e

    VENV_DIR=/root/venv-$MB_PYTHON_TAG

    # Setup a virtual environment for the target python version
    /opt/python/$MB_PYTHON_TAG/bin/python -m pip install pip
    /opt/python/$MB_PYTHON_TAG/bin/python -m pip install setuptools pip virtualenv cmake wheel
    /opt/python/$MB_PYTHON_TAG/bin/python -m virtualenv $VENV_DIR

    source $VENV_DIR/bin/activate 

    cd /io
    pip install -r requirements/build.txt

    CUDA_VERSION=$(ls /usr/local/cuda/lib64/libcudart.so.*|sort|tac | head -1 | rev | cut -d"." -f -3 | rev) # 10.1.243
    CUDA_VERSION_SHORT=$(ls /usr/local/cuda/lib64/libcudart.so.*|sort|tac | head -1 | rev | cut -d"." -f -3 | rev | cut -f1,2 -d".") # 10.1
    CUDNN_VERSION=$(ls /usr/local/cuda/lib64/libcudnn.so.*|sort|tac | head -1 | rev | cut -d"." -f -3 | rev)
    echo "
    CUDA_VERSION = $CUDA_VERSION
    CUDA_VERSION_SHORT = $CUDA_VERSION_SHORT
    CUDNN_VERSION = $CUDNN_VERSION
    "
    
    #TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
    TORCH_CUDA_ARCH_LIST="3.7+PTX;5.0;6.0;6.1;7.0;7.5"
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
    echo "
    TORCH_CUDA_ARCH_LIST = $TORCH_CUDA_ARCH_LIST
    TORCH_NVCC_FLAGS = $TORCH_NVCC_FLAGS
    LD_LIBRARY_PATH = $LD_LIBRARY_PATH
    "
    python setup.py bdist_wheel

    # note that pip install of the unpatched wheel in this environment works

    # does patching need to be done?
    # https://github.com/pytorch/builder/blob/2d91d533eccbe0d859cfe38a9d508fe065f47534/manywheel/build_common.sh 

    chmod -R o+rw _skbuild
    chmod -R o+rw dist
    chmod -R o+rw build
    chmod -R o+rw .eggs

    # Do we have to manually patch the wheels? The pytorch/builder repo
    # mentions that: "auditwheel repair doesnt work correctly and is buggy"
    # perhaps we should look into that?

    /opt/python/cp37-cp37m/bin/python -m pip install auditwheel
    /opt/python/cp37-cp37m/bin/python -m auditwheel show dist/$NAME-*$MB_PYTHON_TAG*.whl
    /opt/python/cp37-cp37m/bin/python -m auditwheel repair --plat=manylinux2014_x86_64 dist/$NAME-*$MB_PYTHON_TAG*.whl 

    #/opt/python/cp37-cp37m/bin/python -m auditwheel show dist/$NAME-$VERSION-$MB_PYTHON_TAG*.whl
    #/opt/python/cp37-cp37m/bin/python -m auditwheel repair dist/$NAME-$VERSION-$MB_PYTHON_TAG*.whl
    chmod -R o+rw wheelhouse
    chmod -R o+rw $NAME.egg-info

    __debug__="""

    python -c 'import mmdet.ops' 
    cd /io/wheelhouse

    """
fi

