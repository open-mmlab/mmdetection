# GET STARTED

## Prerequisites

In this section, we demonstrate how to prepare an environment with PyTorch.

MMDetection works on Linux, Windows, and macOS. It requires Python 3.7+, CUDA 9.2+, and PyTorch 1.6+.

```{note}
If you are experienced with PyTorch and have already installed it, just skip this part and jump to the [next section](#installation). Otherwise, you can follow these steps for the preparation.
```

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

## Installation

We recommend that users follow our best practices to install MMDetection. However, the whole process is highly customizable. See [Customize Installation](#customize-installation) section for more information.

### Best Practices

**Step 0.** Install [MMEngine](https://github.com/open-mmlab/mmengine) and [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

**Note:** In MMCV-v2.x, `mmcv-full` is rename to `mmcv`, if you want to install `mmcv` without CUDA ops, you can use `mim install "mmcv-lite>=2.0.0rc1"` to install the lite version.

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

Case b: If you use mmdet as a dependency or third-party package, install it with MIM:

```shell
mim install mmdet
```

## Verify the installation

To verify whether MMDetection is installed correctly, we provide some sample codes to run an inference demo.

**Step 1.** We need to download config and checkpoint files.

```shell
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
```

The downloading will take several seconds or more, depending on your network environment. When it is done, you will find two files `rtmdet_tiny_8xb32-300e_coco.py` and `rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth` in your current folder.

**Step 2.** Verify the inference demo.

Case a: If you install MMDetection from source, just run the following command.

```shell
python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cpu
```

You will see a new image `demo.jpg` on your `./outputs/vis` folder, where bounding boxes are plotted on cars, benches, etc.

Case b: If you install MMDetection with MIM, open your python interpreter and copy&paste the following codes.

```python
from mmdet.apis import init_detector, inference_detector

config_file = 'rtmdet_tiny_8xb32-300e_coco.py'
checkpoint_file = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
inference_detector(model, 'demo/demo.jpg')
```

You will see a list of `DetDataSample`, and the predictions are in the `pred_instance`, indicating the detected bounding boxes, labels, and scores.

## Tracking Installation

We recommend that users follow our best practices to install MMDetection for for tracking task.

### Best Practices

**Step 0.** Install [MMEngine](https://github.com/open-mmlab/mmengine) and [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

**Step 1.** Install MMDetection.

Case a: If you develop and run mmdet directly, install it from source:

```shell
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e . -r requirements/tracking.txt
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

Case b: If you use mmdet as a dependency or third-party package, install it with MIM:

```shell
mim install mmdet[tracking]
```

**Step 2.** Install TrackEval.

```shell
pip install git+https://github.com/JonathonLuiten/TrackEval.git
```

## Verify the installation

To verify whether MMDetection is installed correctly, we provide some sample codes to run an inference demo.

**Step 1.** We need to download config and checkpoint files.

```shell
mim download mmdet --config bytetrack_yolox_x_8xb4-amp-80e_crowdhuman-mot17halftrain_test-mot17halfval --dest .
```

The downloading will take several seconds or more, depending on your network environment. When it is done, you will find two files `bytetrack_yolox_x_8xb4-amp-80e_crowdhuman-mot17halftrain_test-mot17halfval.py` and `bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth` in your current folder.

**Step 2.** Verify the inference demo.

Case a: If you install MMDetection from source, just run the following command.

```shell
python demo/mot_demo.py demo/demo_mot.mp4 bytetrack_yolox_x_8xb4-amp-80e_crowdhuman-mot17halftrain_test-mot17halfval.py --checkpoint bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth --out mot.mp4
```

You will see a new video `mot.mp4` on your folder, where bounding boxes are plotted on person.

Case b: If you install MMDetection with MIM, open your python interpreter and demo/mot_demo.py, then run it like Case a.

### Customize Installation

#### CUDA versions

When installing PyTorch, you need to specify the version of CUDA. If you are not clear on which to choose, follow our recommendations:

- For Ampere-based NVIDIA GPUs, such as GeForce 30 series and NVIDIA A100, CUDA 11 is a must.
- For older NVIDIA GPUs, CUDA 11 is backward compatible, but CUDA 10.2 offers better compatibility and is more lightweight.

Please make sure the GPU driver satisfies the minimum version requirements. See [this table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions) for more information.

```{note}
Installing CUDA runtime libraries is enough if you follow our best practices, because no CUDA code will be compiled locally. However, if you hope to compile MMCV from source or develop other CUDA operators, you need to install the complete CUDA toolkit from NVIDIA's [website](https://developer.nvidia.com/cuda-downloads), and its version should match the CUDA version of PyTorch. i.e., the specified version of cudatoolkit in the `conda install` command.
```

#### Install MMEngine without MIM

To install MMEngine with pip instead of MIM, please follow [MMEngine installation guides](https://mmengine.readthedocs.io/en/latest/get_started/installation.html).

For example, you can install MMEngine by the following command.

```shell
pip install mmengine
```

#### Install MMCV without MIM

MMCV contains C++ and CUDA extensions, thus depending on PyTorch in a complex way. MIM solves such dependencies automatically and makes the installation easier. However, it is not a must.

To install MMCV with pip instead of MIM, please follow [MMCV installation guides](https://mmcv.readthedocs.io/en/2.x/get_started/installation.html). This requires manually specifying a find-url based on the PyTorch version and its CUDA version.

For example, the following command installs MMCV built for PyTorch 1.12.x and CUDA 11.6.

```shell
pip install "mmcv>=2.0.0" -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html
```

#### Install on CPU-only platforms

MMDetection can be built for CPU-only environments. In CPU mode you can train (requires MMCV version >= 2.0.0rc1), test, or infer a model.

However, some functionalities are gone in this mode:

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

If you try to train/test/infer a model containing the above ops, an error will be raised.
The following table lists affected algorithms.

|                        Operator                         |                                          Model                                           |
| :-----------------------------------------------------: | :--------------------------------------------------------------------------------------: |
| Deformable Convolution/Modulated Deformable Convolution | DCN、Guided Anchoring、RepPoints、CentripetalNet、VFNet、CascadeRPN、NAS-FCOS、DetectoRS |
|                      MaskedConv2d                       |                                     Guided Anchoring                                     |
|                         CARAFE                          |                                          CARAFE                                          |
|                      SyncBatchNorm                      |                                         ResNeSt                                          |

#### Install on Google Colab

[Google Colab](https://colab.research.google.com/) usually has PyTorch installed,
thus we only need to install MMEngine, MMCV, and MMDetection with the following commands.

**Step 1.** Install [MMEngine](https://github.com/open-mmlab/mmengine) and [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
!pip3 install openmim
!mim install mmengine
!mim install "mmcv>=2.0.0,<2.1.0"
```

**Step 2.** Install MMDetection from the source.

```shell
!git clone https://github.com/open-mmlab/mmdetection.git
%cd mmdetection
!pip install -e .
```

**Step 3.** Verification.

```python
import mmdet
print(mmdet.__version__)
# Example output: 3.0.0, or an another version.
```

```{note}
Within Jupyter, the exclamation mark `!` is used to call external executables and `%cd` is a [magic command](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-cd) to change the current working directory of Python.
```

#### Use MMDetection with Docker

We provide a [Dockerfile](../../docker/Dockerfile) to build an image. Ensure that your [docker version](https://docs.docker.com/engine/install/) >=19.03.

```shell
# build an image with PyTorch 1.9, CUDA 11.1
# If you prefer other versions, just modified the Dockerfile
docker build -t mmdetection docker/
```

Run it with

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmdetection/data mmdetection
```

### Troubleshooting

If you have some issues during the installation, please first view the [FAQ](notes/faq.md) page.
You may [open an issue](https://github.com/open-mmlab/mmdetection/issues/new/choose) on GitHub if no solution is found.

### Use Multiple Versions of MMDetection in Development

Training and testing scripts have already been modified in `PYTHONPATH` in order to make sure the scripts are using their own versions of MMDetection.

To install the default version of MMDetection in your environment, you can exclude the follow code in the relative scripts:

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```
