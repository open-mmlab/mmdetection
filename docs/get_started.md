## Prerequisites

- Linux or macOS (Windows is in experimental support)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

The compatible MMDetection and MMCV versions are as below. Please install the correct version of MMCV to avoid installation issues.

| MMDetection version |    MMCV version     |
|:-------------------:|:-------------------:|
| master              | mmcv-full>=1.1.5, <=1.3|
| 2.6.0               | mmcv-full>=1.1.5, <=1.3|
| 2.5.0               | mmcv-full>=1.1.5, <=1.3|
| 2.4.0               | mmcv-full>=1.1.1, <=1.3|
| 2.3.0               | mmcv-full==1.0.5|
| 2.3.0rc0            | mmcv-full>=1.0.2    |
| 2.2.1               | mmcv==0.6.2         |
| 2.2.0               | mmcv==0.6.2         |
| 2.1.0               | mmcv>=0.5.9, <=0.6.1|
| 2.0.0               | mmcv>=0.5.1, <=0.5.8|

Note: You need to run `pip uninstall mmcv` first if you have mmcv installed.
If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.

## Installation

1. Create a conda virtual environment and activate it.

    ```shell
    conda create -n open-mmlab python=3.7 -y
    conda activate open-mmlab
    ```

2. Install PyTorch and torchvision following the [official instructions
](https://pytorch.org/), e.g.,

    ```shell
    conda install pytorch torchvision -c pytorch
    ```

    Note: Make sure that your compilation CUDA version and runtime CUDA version match.
    You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

    `E.g.1` If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install
    PyTorch 1.5, you need to install the prebuilt PyTorch with CUDA 10.1.

    ```shell
    conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
    ```

    `E.g. 2` If you have CUDA 9.2 installed under `/usr/local/cuda` and would like to install
    PyTorch 1.3.1., you need to install the prebuilt PyTorch with CUDA 9.2.

    ```shell
    conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
    ```

    If you build PyTorch from source instead of installing the prebuilt pacakge,
    you can use more CUDA versions such as 9.0.

3. Install mmcv-full, we recommend you to install the pre-build package as below.

    ```shell
    pip install mmcv-full==latest+torch1.6.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html
    ```

    See [here](https://github.com/open-mmlab/mmcv#install-with-pip) for different versions of MMCV compatible to different PyTorch and CUDA versions.
    Optionally you can choose to compile mmcv from source by the following command

    ```shell
    git clone https://github.com/open-mmlab/mmcv.git
    cd mmcv
    MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full will be installed after this step
    cd ..
    ```

    Or directly run

    ```shell
    pip install mmcv-full
    ```

4. Clone the MMDetection repository.

    ```shell
    git clone https://github.com/open-mmlab/mmdetection.git
    cd mmdetection
    ```

5. Install build requirements and then install MMDetection.

    ```shell
    pip install -r requirements/build.txt
    pip install -v -e .  # or "python setup.py develop"
    ```

Note:

a. Following the above instructions, MMDetection is installed on `dev` mode
, any local modifications made to the code will take effect without the need to reinstall it.

b. If you would like to use `opencv-python-headless` instead of `opencv
-python`,
you can install it before installing MMCV.

c. Some dependencies are optional. Simply running `pip install -v -e .` will
 only install the minimum runtime requirements. To use optional dependencies like `albumentations` and `imagecorruptions` either install them manually with `pip install -r requirements/optional.txt` or specify desired extras when calling `pip` (e.g. `pip install -v -e .[optional]`). Valid keys for the extras field are: `all`, `tests`, `build`, and `optional`.

### Install with CPU only

The code can be built for CPU only environment (where CUDA isn't available).

In CPU mode you can run the demo/webcam_demo.py for example.
However some functionality is gone in this mode:

- Deformable Convolution
- Deformable ROI pooling
- CARAFE: Content-Aware ReAssembly of FEatures
- nms_cuda
- sigmoid_focal_loss_cuda

So if you try to run inference with a model containing deformable convolution you will get an error.

### Another option: Docker Image

We provide a [Dockerfile](https://github.com/open-mmlab/mmdetection/blob/master/docker/Dockerfile) to build an image.

```shell
# build an image with PyTorch 1.6, CUDA 10.1
docker build -t mmdetection docker/
```

Run it with

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmdetection/data mmdetection
```

### A from-scratch setup script

Assuming that you already have CUDA 10.1 installed, here is a full script for setting up MMDetection with conda.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit==10.1 -c pytorch -y

# install the latest mmcv
pip install mmcv-full==latest+torch1.6.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html

# install mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .
```

### Developing with multiple MMDetection versions

The train and test scripts already modify the `PYTHONPATH` to ensure the script use the MMDetection in the current directory.

To use the default MMDetection installed in the environment rather than that you are working with, you can remove the following line in those scripts

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```

## Verification

To verify whether MMDetection and the required environment are installed correctly, we can run sample python codes to initialize a detector and inference a demo image:

```python
from mmdet.apis import init_detector, inference_detector

config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, device=device)
# inference the demo image
inference_detector(model, 'demo/demo.jpg')
```

The above code is supposed to run successfully upon you finish the installation.
