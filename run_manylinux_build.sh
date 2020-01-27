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


fname_with_sha256() {
    HASH=$(sha256sum $1 | cut -c1-8)
    DIRNAME=$(dirname $1)
    BASENAME=$(basename $1)
    if [[ $BASENAME == "libnvrtc-builtins.so" ]]; then
        echo $1
    else
        INITNAME=$(echo $BASENAME | cut -f1 -d".")
        ENDNAME=$(echo $BASENAME | cut -f 2- -d".")
        echo "$DIRNAME/$INITNAME-$HASH.$ENDNAME"
    fi
}

make_wheel_record() {
    __heredoc__="""
    make_wheel_record 
    """
    FPATH=$1
    if echo $FPATH | grep RECORD >/dev/null 2>&1; then
        # if the RECORD file, then
        echo "$FPATH,,"
    else
        HASH=$(openssl dgst -sha256 -binary $FPATH | openssl base64 | sed -e 's/+/-/g' | sed -e 's/\//_/g' | sed -e 's/=//g')
        FSIZE=$(ls -nl $FPATH | awk '{print $5}')
        echo "$FPATH,sha256=$HASH,$FSIZE"
    fi
}




custom_repair_wheel(){
    __heredoc__="""
    See:
        https://github.com/pytorch/builder/blob/2d91d533eccbe0d859cfe38a9d508fe065f47534/manywheel/build_common.sh#L226

    Ignore:
        pkg=/io/dist/mmdet-1.0rc1+c73c7e0-cp37-cp37m-linux_x86_64.whl
    """
    pkg=$1   # WHEEL FILEPATH

    # Create temporary work dir
    mkdir -p tmp
    cd tmp
    cp $pkg .

    # Unpack the wheel
    unzip -q $(basename $pkg)
    rm -f $(basename $pkg)
    
    PREFIX=mmdet
    mkdir -p $PREFIX/lib

    export PATCHELF_BIN=/usr/local/bin/patchelf
    patchelf_version=`$PATCHELF_BIN --version`
    echo "patchelf version: " $patchelf_version
    if [[ "$patchelf_version" == "patchelf 0.9" ]]; then
        echo "Your patchelf version is too old. Please use version >= 0.10."
        exit 1
    fi
        

    echo $CUDA_VERSION_SHORT

    OS_NAME=`awk -F= '/^NAME/{print $2}' /etc/os-release`
    if [[ "$OS_NAME" == *"CentOS Linux"* ]]; then
        LIBGOMP_PATH="/usr/lib64/libgomp.so.1"
    elif [[ "$OS_NAME" == *"Ubuntu"* ]]; then
        LIBGOMP_PATH="/usr/lib/x86_64-linux-gnu/libgomp.so.1"
    fi

    if [[ $CUDA_VERSION_SHORT == "9.0" ]]; then
    DEPS_LIST=(
        "/usr/local/cuda/lib64/libcudart.so.9.0"
        "/usr/local/cuda/lib64/libnvToolsExt.so.1"
        "/usr/local/cuda/lib64/libnvrtc.so.9.0"
        "/usr/local/cuda/lib64/libnvrtc-builtins.so"
        "$LIBGOMP_PATH"
    )
    DEPS_SONAME=(
        "libcudart.so.9.0"
        "libnvToolsExt.so.1"
        "libnvrtc.so.9.0"
        "libnvrtc-builtins.so"
        "libgomp.so.1"
    )
    elif [[ $CUDA_VERSION_SHORT == "9.2" ]]; then
    DEPS_LIST=(
        "/usr/local/cuda/lib64/libcudart.so.9.2"
        "/usr/local/cuda/lib64/libnvToolsExt.so.1"
        "/usr/local/cuda/lib64/libnvrtc.so.9.2"
        "/usr/local/cuda/lib64/libnvrtc-builtins.so"
        "$LIBGOMP_PATH"
    )

    DEPS_SONAME=(
        "libcudart.so.9.2"
        "libnvToolsExt.so.1"
        "libnvrtc.so.9.2"
        "libnvrtc-builtins.so"
        "libgomp.so.1"
    )
    elif [[ $CUDA_VERSION_SHORT == "10.0" ]]; then
    DEPS_LIST=(
        "/usr/local/cuda/lib64/libcudart.so.10.0"
        "/usr/local/cuda/lib64/libnvToolsExt.so.1"
        "/usr/local/cuda/lib64/libnvrtc.so.10.0"
        "/usr/local/cuda/lib64/libnvrtc-builtins.so"
        "$LIBGOMP_PATH"
    )

    DEPS_SONAME=(
        "libcudart.so.10.0"
        "libnvToolsExt.so.1"
        "libnvrtc.so.10.0"
        "libnvrtc-builtins.so"
        "libgomp.so.1"
    )
    elif [[ $CUDA_VERSION_SHORT == "10.1" ]]; then
    DEPS_LIST=(
        "/usr/local/cuda/lib64/libcudart.so.10.1"
        "/usr/local/cuda/lib64/libnvToolsExt.so.1"
        "/usr/local/cuda/lib64/libnvrtc.so.10.1"
        "/usr/local/cuda/lib64/libnvrtc-builtins.so"
        "$LIBGOMP_PATH"
    )
    DEPS_SONAME=(
        "libcudart.so.10.1"
        "libnvToolsExt.so.1"
        "libnvrtc.so.10.1"
        "libnvrtc-builtins.so"
        "libgomp.so.1"
    )
    else
        echo "Unknown cuda version $CUDA_VERSION_SHORT"
        exit 1
    fi

    if [[ $pkg != *"without-deps"* ]]; then
        # copy over needed dependent .so files over and tag them with their hash
        patched=()
        for filepath in "${DEPS_LIST[@]}"; do
            filename=$(basename $filepath)
            destpath=$PREFIX/lib/$filename
            if [[ "$filepath" != "$destpath" ]]; then
                cp -v $filepath $destpath
            fi

            patchedpath=$(fname_with_sha256 $destpath)
            patchedname=$(basename $patchedpath)
            if [[ "$destpath" != "$patchedpath" ]]; then
                mv $destpath $patchedpath
            fi
            patched+=("$patchedname")
            echo "Copied $filepath to $patchedpath"
        done

        echo "patching to fix the so names to the hashed names"
        for ((i=0;i<${#DEPS_LIST[@]};++i)); do
            find $PREFIX -name '*.so*' | while read sofile; do
                origname=${DEPS_SONAME[i]}
                patchedname=${patched[i]}
                if [[ "$origname" != "$patchedname" ]]; then
                    set +e
                    $PATCHELF_BIN --print-needed $sofile | grep $origname 2>&1 >/dev/null
                    ERRCODE=$?
                    set -e
                    if [ "$ERRCODE" -eq "0" ]; then
                        echo "patching $sofile entry $origname to $patchedname"
                        $PATCHELF_BIN --replace-needed $origname $patchedname $sofile
                    fi
                fi
            done
        done
    fi
    
    # set RPATH of _C.so and similar to $ORIGIN, $ORIGIN/lib
    find $PREFIX -maxdepth 1 -type f -name "*.so*" | while read sofile; do
        echo "Setting rpath of $sofile to " '$ORIGIN:$ORIGIN/lib'
        $PATCHELF_BIN --set-rpath '$ORIGIN:$ORIGIN/lib' $sofile
        $PATCHELF_BIN --print-rpath $sofile
    done

    # set RPATH of lib/ files to $ORIGIN
    find $PREFIX/lib -maxdepth 1 -type f -name "*.so*" | while read sofile; do
        echo "Setting rpath of $sofile to " '$ORIGIN'
        $PATCHELF_BIN --set-rpath '$ORIGIN' $sofile
        $PATCHELF_BIN --print-rpath $sofile
    done

    # regenerate the RECORD file with new hashes
    record_file=`echo $(basename $pkg) | sed -e 's/-cp.*$/.dist-info\/RECORD/g'`
    if [[ -e $record_file ]]; then
        echo "Generating new record file $record_file"
        rm -f $record_file
        # generate records for folders in wheel
        find * -type f | while read fname; do
            echo $(make_wheel_record $fname) >>$record_file
        done
    fi

    # zip up the wheel back
    zip -rq $(basename $pkg) $PREIX*

    mkdir -p ../custom_wheelhouse
    mv $(basename $pkg) ../custom_wheelhouse/$(basename $pkg)
    cd ..
    rm -rf tmp
}

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

    yum install mod_ssl zip -y

    VENV_DIR=/root/venv-$MB_PYTHON_TAG
    # Setup a virtual environment for the target python version
    /opt/python/$MB_PYTHON_TAG/bin/python -m pip install pip -U
    /opt/python/$MB_PYTHON_TAG/bin/python -m pip install virtualenv
    /opt/python/$MB_PYTHON_TAG/bin/python -m virtualenv $VENV_DIR

    source $VENV_DIR/bin/activate 

    cd /io
    pip pip -U
    pip install scikit-build
    pip install cmake
    pip install ninja
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
    #TORCH_CUDA_ARCH_LIST="3.7+PTX;5.0;6.0;6.1;7.0;7.5"
    TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5"
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
    #export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
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

    TORCH_DPATH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))")
    export LD_LIBRARY_PATH="$TORCH_DPATH/lib:$LD_LIBRARY_PATH"

    custom_repair_wheel dist/$NAME-*$MB_PYTHON_TAG*.whl

    /opt/python/cp37-cp37m/bin/python -m pip install auditwheel
    /opt/python/cp37-cp37m/bin/python -m auditwheel show dist/$NAME-*$MB_PYTHON_TAG*.whl
    /opt/python/cp37-cp37m/bin/python -m auditwheel show custom_wheelhouse/$NAME-*$MB_PYTHON_TAG*.whl
    /opt/python/cp37-cp37m/bin/python -m auditwheel repair --plat=manylinux2014_x86_64 dist/$NAME-*$MB_PYTHON_TAG*.whl 

    #/opt/python/cp37-cp37m/bin/python -m auditwheel show dist/$NAME-$VERSION-$MB_PYTHON_TAG*.whl
    #/opt/python/cp37-cp37m/bin/python -m auditwheel repair dist/$NAME-$VERSION-$MB_PYTHON_TAG*.whl
    chmod -R o+rw wheelhouse
    chmod -R o+rw $NAME.egg-info

    __debug__="""

    python -c 'import mmdet.ops' 
    cd /io/wheelhouse

    pip install -r /io/requirements.txt

    pip install mmdet.*.whl

    cd /io/_skbuild/linux-x86_64-3.7/cmake-install/mmdet/ops/nms
    python -c 'import torch; import nms_cuda' 

    python -c 'import mmdet.ops' 
    python -c 'import soft_nms_cpu' 

    find . -iname 'lib*.so' -delete

    conda create -n test37 python=3.7

    find /usr/local/ -iname 'libc10*'
    find $TORCH_DPATH -iname 'libc10*'

    """
fi

