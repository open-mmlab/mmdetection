#!/usr/bin/env bash

work_dir=$(realpath "$(dirname $0)")

venv_dir=$1
if [ -z "$venv_dir" ]; then
  venv_dir=venv
fi

cd ${work_dir}

if [[ -e ${venv_dir} ]]; then
  echo
  echo "Virtualenv already exists. Use command to start working:"
  echo "$ . ${venv_dir}/bin/activate"
  exit
fi

# Create virtual environment
python3 -m venv ${venv_dir} --prompt="detection"

. ${venv_dir}/bin/activate

# Get CUDA version.
CUDA_HOME_CANDIDATE=/usr/local/cuda
if [ -z "${CUDA_HOME}" ] && [ -d ${CUDA_HOME_CANDIDATE} ]; then
  echo "Exporting CUDA_HOME as ${CUDA_HOME_CANDIDATE}"
  export CUDA_HOME=${CUDA_HOME_CANDIDATE}
fi

if [ -e "$CUDA_HOME" ]; then
  if [ -e "$CUDA_HOME/version.txt" ]; then
    # Get CUDA version from version.txt file.
    CUDA_VERSION=$(cat $CUDA_HOME/version.txt | sed -e "s/^.*CUDA Version *//" -e "s/ .*//")
  else
    # Get CUDA version from directory name.
    CUDA_HOME_DIR=`readlink -f $CUDA_HOME`
    CUDA_HOME_DIR=`basename $CUDA_HOME_DIR`
    CUDA_VERSION=`echo $CUDA_HOME_DIR | cut -d "-" -f 2`
  fi
fi

if [[ -z ${CUDA_VERSION} ]]; then
  echo "CUDA was not found, installing dependencies in CPU-only mode. If you want to use CUDA, set CUDA_HOME and CUDA_VERSION beforehand."
else
  echo "Using CUDA_VERSION ${CUDA_VERSION}"
fi

# Remove dots from CUDA version string, if any.
CUDA_VERSION_CODE=$(echo ${CUDA_VERSION} | sed -e "s/\.//" -e "s/\(...\).*/\1/")

# install PyTorch and MMCV.
export TORCH_VERSION=1.8.1
export TORCHVISION_VERSION=0.9.1
export NUMPY_VERSION=1.19.5
export MMCV_VERSION=1.3.0

pip install wheel

CONSTRAINTS_FILE=$(tempfile)
echo numpy==${NUMPY_VERSION} >> ${CONSTRAINTS_FILE}

if [[ -z $CUDA_VERSION_CODE ]]; then
  pip install torch==${TORCH_VERSION}+cpu torchvision==${TORCHVISION_VERSION}+cpu -f https://download.pytorch.org/whl/torch_stable.html
  echo torch==${TORCH_VERSION}+cpu >> ${CONSTRAINTS_FILE}
  echo torchvision==${TORCHVISION_VERSION}+cpu >> ${CONSTRAINTS_FILE}
elif [[ $CUDA_VERSION_CODE == "102" ]]; then
  pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION}
  echo torch==${TORCH_VERSION} >> ${CONSTRAINTS_FILE}
  echo torchvision==${TORCHVISION_VERSION} >> ${CONSTRAINTS_FILE}
else
  pip install torch==${TORCH_VERSION}+cu${CUDA_VERSION_CODE} torchvision==${TORCHVISION_VERSION}+cu${CUDA_VERSION_CODE} -f https://download.pytorch.org/whl/torch_stable.html
  echo torch==${TORCH_VERSION}+cu${CUDA_VERSION_CODE} >> ${CONSTRAINTS_FILE}
  echo torchvision==${TORCHVISION_VERSION}+cu${CUDA_VERSION_CODE} >> ${CONSTRAINTS_FILE}
fi

if [[ -z $CUDA_VERSION_CODE ]]; then
  pip install --no-cache-dir mmcv-full==${MMCV_VERSION} -f https://download.openmmlab.com/mmcv/dist/cpu/torch${TORCH_VERSION}/index.html -c ${CONSTRAINTS_FILE}
else
  pip install --no-cache-dir mmcv-full==${MMCV_VERSION} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA_VERSION_CODE}/torch${TORCH_VERSION}/index.html -c ${CONSTRAINTS_FILE}
fi

# Install other requirements.
# Install mmpycocotools and Polygon3 from source to make sure it is compatible with installed numpy version.
pip install --no-cache-dir --no-binary=mmpycocotools mmpycocotools -c ${CONSTRAINTS_FILE}
pip install --no-cache-dir --no-binary=Polygon3 Polygon3==3.0.8 -c ${CONSTRAINTS_FILE}
cat requirements.txt | xargs -n 1 -L 1 pip install --no-cache -c ${CONSTRAINTS_FILE}

pip install -e . -c ${CONSTRAINTS_FILE}
MMDETECTION_DIR=`realpath .`
echo "export MMDETECTION_DIR=${MMDETECTION_DIR}" >> ${venv_dir}/bin/activate

deactivate

echo
echo "Activate a virtual environment to start working:"
echo "$ . ${venv_dir}/bin/activate"
