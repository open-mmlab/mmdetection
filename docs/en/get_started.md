# Prerequisites

In this section we desmonstrate how to prepare an environment with PyTorch.

MMDetection works on Linux, Windows and macOS. It requires Python 3.6+, CUDA 9.2+ and PyTorch 1.5+.

:::{note}
If you are experienced with PyTorch and have already installed it, just skip this part and jump to the [next section](#installation). Otherwise, you can follow these steps for the preparation.
:::

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```shell
conda install pytorch torchvision -c pytorch
```

On CPU platforms:

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

# Installation

We recommend that users follow our best practices to install MMDetetion. However, the whole process is highly customizable. See [Customize Installation](#customize-installation) section for more information.

## Best Practices

**Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmcv-full
```

**Step 1.** Install MMDetection.

Case a: If you develop and run mmdet directly, install it from source:

```shell
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

Case b: If you use mmdet as a dependency or third-party package, install it with pip:

```shell
pip install mmdet
```

## Verify the installation

To verify whether MMDetection is installed correctly, we provide some sample codes to run a inference demo.

**Step 1.** We need to download config and checkpoint files.

```shell
mim download mmdet --config yolov3_mobilenetv2_320_300e_coco --dest .
```

The downloading will take several seconds or more, depending on your network environment. When it is done, you will find two files `yolov3_mobilenetv2_320_300e_coco.py` and `yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth` in your current folder.

**Step 2.** Verify the inference demo.

Option (a). If you install mmdetection from source, just run the following command.

```shell
python demo/image_demo.py demo/demo.jpg yolov3_mobilenetv2_320_300e_coco.py yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth --device cpu --out-file result.jpg
```

You will see a new image `result.jpg` on your current folder, where a bounding box will be shown on the cat.

Option (b). If you install mmdetection with pip, open you python interpreter and copy&paste the following codes.

```python
from mmdet.apis import init_detector, inference_detector

config_file = 'configs/yolo/yolov3_mobilenetv2_320_300e_coco.py'
checkpoint_file = 'yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
inference_detector(model, 'demo/cat.jpg')
```

You will see a list of arrays printed, indicating the detected bounding boxes.

## Customize Installation

### CUDA versions

When installing PyTorch, you need to specify the version of CUDA. If you are not clear on which to choose, follow our recommendations:

- For Ampere-based NVIDIA GPUs, such as GeForce 30 series and NVIDIA A100, CUDA 11 is a must.
- For older NVIDIA GPUs, CUDA 11 is backward compatible, but CUDA 10.2 offers better compatibility and is more lightweight.

Please Note that there is no need to install the complete CUDA toolkit if you follow our best practices, because no CUDA code will be compiled. However if you hope to compile MMCV or other CUDA operators, you need to install the complete CUDA toolkit from NVIDIA's [website](https://developer.nvidia.com/cuda-downloads), and its version should match the CUDA version of PyTorch. i.e., the specified version of cudatoolkit in `conda install` command.

Please also make sure the GPU driver satisfies the minimum version requirements. See [this table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions) for more information.

### Install MMCV without MIM

MMCV contains C++ and CUDA extensions, thus depending on PyTorch in a complex way. MIM solves such dependencies automatically and makes the installation easier. However, it is not a must.

To install MMCV with pip instead of MIM, please follow [MMCV installation guides](https://mmcv.readthedocs.io/en/latest/get_started/installation.html). This requires manually specifying a find-url based on PyTorch version and its CUDA version.

For example, the following command install mmcv-full built for PyTorch 1.10.x and CUDA 11.3.

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

### Install on CPU-only platforms

MMDetection can be built for CPU only environment. In CPU mode you can train (requires MMCV version >= 1.4.4), test or inference a model.

However some functionalities are gone in this mode:

- Deformable Convolution
- Modulated Deformable Convolution
- ROI pooling
- Deformable ROI pooling
- CARAFE
- SyncBatchNorm
- CrissCrossAttention
- MaskedConv2d
- Temporal Interlace Shift
- nms_cuda
- sigmoid_focal_loss_cuda
- bbox_overlaps

If you try to train/test/inference a model containing above ops, an error will be raised.
The following table lists affected algorithms.

|                        Operator                         |                            Model                             |
| :-----------------------------------------------------: | :----------------------------------------------------------: |
| Deformable Convolution/Modulated Deformable Convolution | DCN、Guided Anchoring、RepPoints、CentripetalNet、VFNet、CascadeRPN、NAS-FCOS、DetectoRS |
|                      MaskedConv2d                       |                       Guided Anchoring                       |
|                         CARAFE                          |                            CARAFE                            |
|                      SyncBatchNorm                      |                           ResNeSt                            |

### Install on Google Colab

[Google Colab](https://research.google.com/) usually has PyTorch installed,
thus we only need to install MMCV and MMDetection with the following commands.

**Step 1.**
Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
!pip3 install openmim
!mim install mmcv-full
```

**Step 2.**
Install MMDetection from the source.

```shell
!git clone https://github.com/open-mmlab/mmdetection.git
%cd mmdetection
!pip install -e .
```

**Step 3.**
Verification.

```python
import mmdet
print(mmdet.__version__)
# Example output: 2.23.0
```

```{note}
Within Jupyter, the exclamation mark `!` is used to call external executables and `%cd` is a [magic command](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-cd) to change the current working directory of Python.
```

### Using MMDetection with Docker

We provide a [Dockerfile](https://github.com/open-mmlab/mmdetection/blob/master/docker/Dockerfile) to build an image. Ensure that you are using [docker version](https://docs.docker.com/engine/install/) >=19.03.

```shell
# build an image with PyTorch 1.6, CUDA 10.1
# If you prefer other versions, just modified the Dockerfile
docker build -t mmdetection docker/
```

Run it with

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmdetection/data mmdetection
```

## FAQ

### Install albumentations

If you would like to use `albumentations`, we suggest using `pip install -r requirements/albu.txt` or
`pip install -U albumentations --no-binary qudida,albumentations`.
If you simply use `pip install albumentations>=0.3.2`, it will install `opencv-python-headless` simultaneously (even though you have already installed `opencv-python`).
Please refer to the [official documentation](https://albumentations.ai/docs/getting_started/installation/#note-on-opencv-dependencies) for details.

### ModuleNotFoundError is raised when using some algorithms

Some extra dependencies are required for Instaboost, Panoptic Segmentation, LVIS dataset, etc. Please note the error message and install corresponding packages, e.g.,

```shell
# for instaboost
pip install instaboostfast
# for panoptic segmentation
pip install git+https://github.com/cocodataset/panopticapi.git
# for LVIS dataset
pip install git+https://github.com/lvis-dataset/lvis-api.git
```

### Do I need to reinstall mmdet after some code modifications

If you follow the best practice and install mmdet with `pip install -e .`, any local modifications made to the code will take effect without reinstallation.

b. If you would like to use `opencv-python-headless` instead of `opencv-python`,
you can install it before installing MMCV.

### How to develop with multiple MMDetection versions

You can have multiple folders like mmdet-2.21, mmdet-2.22.
When you run the train or test script, it will adopt the mmdet package in the current folder.

To use the default MMDetection installed in the environment rather than the one you are working with, you can remove the following line in those scripts:

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```

### ModuleNotFoundError with `mmcv._ext`

If you have installed mmcv instead of mmcv-full before, please run `pip uninstall mmcv` first and then install mmcv-full.

### Compatibility between MMCV and MMDetection

Compatible MMDetection and MMCV versions are shown as below. Please choose the correct version of MMCV to avoid installation issues.

| MMDetection version |       MMCV version        |
|:-------------------:|:-------------------------:|
|       master        | mmcv-full>=1.3.17, <1.6.0 |
|       2.24.0        | mmcv-full>=1.3.17, <1.6.0 |
|       2.23.0        | mmcv-full>=1.3.17, <1.5.0 |
|       2.22.0        | mmcv-full>=1.3.17, <1.5.0 |
|       2.21.0        | mmcv-full>=1.3.17, <1.5.0 |
|       2.20.0        | mmcv-full>=1.3.17, <1.5.0 |
|       2.19.1        | mmcv-full>=1.3.17, <1.5.0 |
|       2.19.0        | mmcv-full>=1.3.17, <1.5.0 |
|       2.18.0        | mmcv-full>=1.3.17, <1.4.0 |
|       2.17.0        | mmcv-full>=1.3.14, <1.4.0 |
|       2.16.0        | mmcv-full>=1.3.8, <1.4.0  |
|       2.15.1        | mmcv-full>=1.3.8, <1.4.0  |
|       2.15.0        | mmcv-full>=1.3.8, <1.4.0  |
|       2.14.0        | mmcv-full>=1.3.8, <1.4.0  |
|       2.13.0        | mmcv-full>=1.3.3, <1.4.0  |
|       2.12.0        | mmcv-full>=1.3.3, <1.4.0  |
|       2.11.0        | mmcv-full>=1.2.4, <1.4.0  |
|       2.10.0        | mmcv-full>=1.2.4, <1.4.0  |
|        2.9.0        | mmcv-full>=1.2.4, <1.4.0  |
|        2.8.0        | mmcv-full>=1.2.4, <1.4.0  |
|        2.7.0        | mmcv-full>=1.1.5, <1.4.0  |
|        2.6.0        | mmcv-full>=1.1.5, <1.4.0  |
|        2.5.0        | mmcv-full>=1.1.5, <1.4.0  |
|        2.4.0        | mmcv-full>=1.1.1, <1.4.0  |
|        2.3.0        |     mmcv-full==1.0.5      |
|      2.3.0rc0       |     mmcv-full>=1.0.2      |
|        2.2.1        |        mmcv==0.6.2        |
|        2.2.0        |        mmcv==0.6.2        |
|        2.1.0        |   mmcv>=0.5.9, <=0.6.1    |
|        2.0.0        |   mmcv>=0.5.1, <=0.5.8    |
