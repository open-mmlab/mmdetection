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

    cd ~/code/mmdetection/docker
    docker build --build-arg BASE_IMAGE=pytorch/manylinux-cuda101 -t mmdet/manylinux-cuda101 -f Dockerfile_base .
    docker build --build-arg BASE_IMAGE=pytorch/manylinux-cuda92 -t mmdet/manylinux-cuda92 -f Dockerfile_base .

"""


#DOCKER_IMAGE=${DOCKER_IMAGE:="quay.io/pypa/manylinux2010_x86_64:latest"}
#DOCKER_IMAGE=${DOCKER_IMAGE:="soumith/manylinux-cuda101:latest"}
DOCKER_IMAGE=${DOCKER_IMAGE:="mmdet/manylinux-cuda101"}

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
        -e HOST_USER="$USER" \
        -e VERSION="$VERSION" \
        $DOCKER_IMAGE bash -c 'cd /io && ./run_manylinux_build.sh'

    __interactive__='''
    # notes for running / debugging interactively 

    docker run --runtime=nvidia --rm \
        -v $PWD:/io \
        -e _INSIDE_DOCKER="YES" \
        -e MB_PYTHON_TAG="$MB_PYTHON_TAG" \
        -e NAME="$NAME" \
        -e HOST_USER="$USER" \
        -e VERSION="$VERSION" \
        -it $DOCKER_IMAGE bash

    set +e
    set +x
    '''

    BDIST_WHEEL_PATH=$(ls custom_wheelhouse/$NAME-$VERSION-$MB_PYTHON_TAG*.whl)
    echo "BDIST_WHEEL_PATH = $BDIST_WHEEL_PATH"
fi
    #set -x
    #set -e


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
        whl_fpath=/io/dist/mmdet-1.0rc1+c73c7e0-cp37-cp37m-linux_x86_64.whl
    """
    whl_fpath=$1   # WHEEL FILEPATH

    echo "repair whl_fpath = $whl_fpath"

    # Create temporary work dir
    prev_dpath=$(pwd)
    tmp_dpath=$prev_dpath/tmp

    mkdir -p $tmp_dpath
    cd $tmp_dpath
    cp $whl_fpath .

    # Unpack the wheel
    unzip -q $(basename $whl_fpath)
    rm -f $(basename $whl_fpath)
    
    PREFIX=mmdet
    mkdir -p $PREFIX/lib

    export PATCHELF_BIN=/usr/local/bin/patchelf
    patchelf_version=`$PATCHELF_BIN --version`
    echo "patchelf version: " $patchelf_version
    if [[ "$patchelf_version" == "patchelf 0.9" ]]; then
        echo "Your patchelf version is too old. Please use version >= 0.10."
        exit 1
    fi
        
    CUDA_VERSION_FULL=$(ls /usr/local/cuda/lib64/libcudart.so.*|sort|tac | head -1 | rev | cut -d"." -f -3 | rev) # 10.1.243
    CUDA_VERSION_SHORT=$(ls /usr/local/cuda/lib64/libcudart.so.*|sort|tac | head -1 | rev | cut -d"." -f -3 | rev | cut -f1,2 -d".") # 10.1
    CUDNN_VERSION=$(ls /usr/local/cuda/lib64/libcudnn.so.*|sort|tac | head -1 | rev | cut -d"." -f -3 | rev)
    echo "
    CUDA_VERSION_FULL = $CUDA_VERSION_FULL
    CUDA_VERSION_SHORT = $CUDA_VERSION_SHORT
    CUDNN_VERSION = $CUDNN_VERSION
    "

    OS_NAME=`awk -F= '/^NAME/{print $2}' /etc/os-release`
    if [[ "$OS_NAME" == *"CentOS Linux"* ]]; then
        LIBGOMP_PATH="/usr/lib64/libgomp.so.1"
    elif [[ "$OS_NAME" == *"Ubuntu"* ]]; then
        LIBGOMP_PATH="/usr/lib/x86_64-linux-gnu/libgomp.so.1"
    fi

    __debug__="""
	linux-vdso.so.1                     =>  (0x00007ffdd7fbf000)
	libcudart.so.10.1                   => /usr/local/cuda/lib64/libcudart.so.10.1 (0x00007f2c190f6000)

	libtorch.so                         => /root/venv-cp37-cp37m/lib/python3.7/site-packages/torch/lib/libtorch.so (0x00007f2bceb55000)
	libc10.so                           => /root/venv-cp37-cp37m/lib/python3.7/site-packages/torch/lib/libc10.so (0x00007f2bce900000)
	libc10_cuda.so                      => /root/venv-cp37-cp37m/lib/python3.7/site-packages/torch/lib/libc10_cuda.so (0x00007f2bcbbca000)
	libcudart-1b201d85.so.10.1          => /root/venv-cp37-cp37m/lib/python3.7/site-packages/torch/lib/libcudart-1b201d85.so.10.1 (0x00007f2ba09b8000)
	libgomp-7c85b1e2.so.1               => /root/venv-cp37-cp37m/lib/python3.7/site-packages/torch/lib/libgomp-7c85b1e2.so.1 (0x00007f2ba078e000)
	libnvToolsExt-3965bdd0.so.1         => /root/venv-cp37-cp37m/lib/python3.7/site-packages/torch/lib/libnvToolsExt-3965bdd0.so.1 (0x00007f2ba0282000)

	libcuda.so.1                        => /usr/lib64/libcuda.so.1 (0x00007f2bcd770000)

	libnvrtc.so.10.1                    => /usr/local/cuda/lib64/libnvrtc.so.10.1 (0x00007f2bcc000000)
	libnvToolsExt.so.1                  => /usr/local/cuda/lib64/libnvToolsExt.so.1 (0x00007f2bcbdf7000)
	libcufft.so.10                      => /usr/local/cuda/lib64/libcufft.so.10 (0x00007f2bc3374000)
	libcurand.so.10                     => /usr/local/cuda/lib64/libcurand.so.10 (0x00007f2bbf313000)
	libcublas.so.10                     => /usr/local/cuda/lib64/libcublas.so.10 (0x00007f2bbb578000)
	libcublasLt.so.10                   => /usr/local/cuda/lib64/libcublasLt.so.10 (0x00007f2b9e191000)

	libcudnn.so.7                       => /usr/local/cuda/lib64/libcudnn.so.7 (0x00007f2ba192e000)

	libnvidia-fatbinaryloader.so.435.21 => /usr/lib64/libnvidia-fatbinaryloader.so.435.21 (0x00007f2ba0035000)

	libpthread.so.0                     => /usr/lib64/libpthread.so.0 (0x00007f2bcb9ae000)
	libstdc++.so.6                      => /usr/lib64/libstdc++.so.6 (0x00007f2ba1627000)
	libgcc_s.so.1                       => /usr/lib64/libgcc_s.so.1 (0x00007f2ba1411000)
	libc.so.6                           => /usr/lib64/libc.so.6 (0x00007f2ba1043000)
	libdl.so.2                          => /usr/lib64/libdl.so.2 (0x00007f2ba0e3f000)
	librt.so.1                          => /usr/lib64/librt.so.1 (0x00007f2ba0c37000)
	libm.so.6                           => /usr/lib64/libm.so.6 (0x00007f2ba048c000)

    ldd mmdet/ops/dcn/deform_conv_cuda.cpython-37m-x86_64-linux-gnu.so
    """

    if [[ $CUDA_VERSION_SHORT == "9.0" ]]; then
        DEPS_LIST=(
            "/usr/local/cuda/lib64/libcudart.so.9.0" 
            "/usr/local/cuda/lib64/libnvToolsExt.so.1" 
            "/usr/local/cuda/lib64/libnvrtc.so.9.0" 
            "/usr/local/cuda/lib64/libnvrtc-builtins.so" "$LIBGOMP_PATH" 
        )
    elif [[ $CUDA_VERSION_SHORT == "9.2" ]]; then
        DEPS_LIST=(
            "/usr/local/cuda/lib64/libcudart.so.9.2" 
            "/usr/local/cuda/lib64/libnvToolsExt.so.1" 
            "/usr/local/cuda/lib64/libnvrtc.so.9.2" 
            "/usr/local/cuda/lib64/libnvrtc-builtins.so" "$LIBGOMP_PATH"
        )
    elif [[ $CUDA_VERSION_SHORT == "10.0" ]]; then
        DEPS_LIST=(
            "/usr/local/cuda/lib64/libcudart.so.10.0" 
            "/usr/local/cuda/lib64/libnvToolsExt.so.1" 
            "/usr/local/cuda/lib64/libnvrtc.so.10.0" 
            "/usr/local/cuda/lib64/libnvrtc-builtins.so" 
            "$LIBGOMP_PATH"
        )
    elif [[ $CUDA_VERSION_SHORT == "10.1" ]]; then
        DEPS_LIST=(
            "/usr/local/cuda/lib64/libcudart.so.10.1" 
            "/usr/local/cuda/lib64/libnvToolsExt.so.1" 
            "/usr/local/cuda/lib64/libnvrtc.so.10.1" 
            "/usr/local/cuda/lib64/libnvrtc-builtins.so" 
            "$LIBGOMP_PATH"
        )
    elif [[ $CUDA_VERSION_SHORT == "10.1" ]]; then
        DEPS_LIST=(
            "/usr/local/cuda/lib64/libcudart.so.10.1" 
            "/usr/local/cuda/lib64/libnvToolsExt.so.1" 
            "/usr/local/cuda/lib64/libnvrtc.so.10.1" 
            "/usr/local/cuda/lib64/libnvrtc-builtins.so" 
            "$LIBGOMP_PATH"
        )
    else
        echo "Unknown cuda version $CUDA_VERSION_SHORT"
        exit 1
    fi

    if [[ $whl_fpath != *"without-deps"* ]]; then
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
            orig_fpath=${DEPS_LIST[i]}
            origname=$(basename $orig_fpath)

            find $PREFIX -name '*.so*' | while read sofile; do

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
    
    # set RPATH of mmdet shared objects to $ORIGIN, $ORIGIN/lib
    find $PREFIX -maxdepth 9 -type f -name "*.so*" | while read sofile; do
        rel_path=$(python -c "import os; print(os.path.relpath('$PREFIX/lib', os.path.dirname('$sofile')))")
        new_rpath="\$ORIGIN:\$ORIGIN/$rel_path"
        echo "Setting rpath of $sofile to " "$new_rpath"
        $PATCHELF_BIN --set-rpath "$new_rpath" $sofile
        $PATCHELF_BIN --print-rpath $sofile
    done

    # set RPATH of lib/ files to $ORIGIN
    find $PREFIX/lib -maxdepth 1 -type f -name "*.so*" | while read sofile; do
        echo "Setting rpath of $sofile to " '$ORIGIN'
        $PATCHELF_BIN --set-rpath '$ORIGIN' $sofile
        $PATCHELF_BIN --print-rpath $sofile
    done

    # regenerate the RECORD file with new hashes
    record_file=`echo $(basename $whl_fpath) | sed -e 's/-cp.*$/.dist-info\/RECORD/g'`
    if [[ -e $record_file ]]; then
        echo "Generating new record file $record_file"
        rm -f $record_file
        # generate records for folders in wheel
        find * -type f | while read fname; do
            echo $(make_wheel_record $fname) >>$record_file
        done
    fi

    # zip up the wheel back
    zip -rq $(basename $whl_fpath) $PREIX*

    mkdir -p ../custom_wheelhouse
    mv $(basename $whl_fpath) ../custom_wheelhouse/$(basename $whl_fpath)

    cd $prev_dpath
    rm -rf $tmp_dpath
}


if [ "$_INSIDE_DOCKER" == "YES" ]; then

    if [ "$(which openssl)" == "" ]; then
        yum install mod_ssl zip -y
    fi

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
    

    # https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility
    #TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
    #TORCH_CUDA_ARCH_LIST="3.7+PTX;5.0;6.0;6.1;7.0;7.5"
    #TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5"

    CUDA_VERSION_FULL=$(ls /usr/local/cuda/lib64/libcudart.so.*|sort|tac | head -1 | rev | cut -d"." -f -3 | rev) # 10.1.243
    CUDA_VERSION_SHORT=$(ls /usr/local/cuda/lib64/libcudart.so.*|sort|tac | head -1 | rev | cut -d"." -f -3 | rev | cut -f1,2 -d".") # 10.1
    CUDNN_VERSION=$(ls /usr/local/cuda/lib64/libcudnn.so.*|sort|tac | head -1 | rev | cut -d"." -f -3 | rev)
    echo "
    CUDA_VERSION_FULL = $CUDA_VERSION_FULL
    CUDA_VERSION_SHORT = $CUDA_VERSION_SHORT
    CUDNN_VERSION = $CUDNN_VERSION
    "

    # Rferences: https://github.com/pytorch/builder/blob/master/manywheel/build.sh
    export TORCH_CUDA_ARCH_LIST="3.7+PTX;5.0"
    if [[ $CUDA_VERSION_SHORT == "9.0" ]]; then
        export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST;6.0;7.0"
    elif [[ $CUDA_VERSION_SHORT == "9.2" ]]; then
        export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST;6.0;6.1;7.0"
    elif [[ $CUDA_VERSION_SHORT == "10.0" ]]; then
        export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST;6.0;6.1;7.0;7.5"
    elif [[ $CUDA_VERSION_SHORT == "10.1" ]]; then
        export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST;6.0;6.1;7.0;7.5"
    else
        echo "unknown cuda version $CUDA_VERSION_SHORT"
        exit 1
    fi
    export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
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

    whl_fpath=$(ls /io/dist/$NAME-*$MB_PYTHON_TAG*.whl)
    echo "whl_fpath = $whl_fpath"

    # Do we have to manually patch the wheels? The pytorch/builder repo
    # mentions that: "auditwheel repair doesnt work correctly and is buggy"
    # perhaps we should look into that?

    #TORCH_DPATH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))")
    #export LD_LIBRARY_PATH="$TORCH_DPATH/lib:$LD_LIBRARY_PATH"

    custom_repair_wheel $whl_fpath

    #/opt/python/cp37-cp37m/bin/python -m pip install auditwheel
    #/opt/python/cp37-cp37m/bin/python -m auditwheel show dist/$NAME-*$MB_PYTHON_TAG*.whl
    #/opt/python/cp37-cp37m/bin/python -m auditwheel show custom_wheelhouse/$NAME-*$MB_PYTHON_TAG*.whl
    #/opt/python/cp37-cp37m/bin/python -m auditwheel repair --plat=manylinux2014_x86_64 dist/$NAME-*$MB_PYTHON_TAG*.whl 

    #/opt/python/cp37-cp37m/bin/python -m auditwheel show dist/$NAME-$VERSION-$MB_PYTHON_TAG*.whl
    #/opt/python/cp37-cp37m/bin/python -m auditwheel repair dist/$NAME-$VERSION-$MB_PYTHON_TAG*.whl
    chmod -R o+rw custom_wheelhouse

    if [ "$HOST_USER" != "" ]; then 
        sudo chown $HOST_USER:$HOST_USER *
    fi
    #chmod -R o+rw wheelhouse
    #chmod -R o+rw *

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

