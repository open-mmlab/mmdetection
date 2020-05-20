## Installation

### Prerequisites

Make sure that all of the listed below system-level requirements are installed:

- Linux (Windows is not officially supported)
- Python 3.6+
- CUDA 9.0 or higher
- NCCL 2
- GCC 4.9 or higher
- [Intel® Distribution of OpenVINO™ Toolkit](https://software.intel.com/en-us/openvino-toolkit) 2020.2

### Install OTEDetection

a. Create a conda virtual environment and activate it.

```shell
conda create -n ote-det python=3.7 -y
conda activate ote-det
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
```

c. Clone OTEDetection repository.

```shell
git clone https://github.com/opencv/mmdetection.git
cd mmdetection
```

d. Install build requirements and then install OTEDetection.

```shell
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

If you build mmdetection on macOS, replace the last command with

```
CC=clang CXX=clang++ CFLAGS='-stdlib=libc++' pip install -e .
```

Note:

1. The git commit id will be written to the version number with step d, e.g. 0.6.0+2e7045c. The version will also be saved in trained models.
It is recommended that you run step d each time you pull some updates from github. If C++/CUDA codes are modified, then this step is compulsory.

2. Following the above instructions, OTEDetection is installed in `dev` mode, any local modifications made to the code will take effect without the need to reinstall it (unless you submit some commits and want to update the version number).

3. If you would like to use `opencv-python-headless` instead of `opencv-python`,
you can install it before installing MMCV.

4. Some dependencies are optional. Simply running `pip install -v -e .` will only install the minimum runtime requirements. To use optional dependencies like `albumentations` and `imagecorruptions` either install them manually with `pip install -r requirements/optional.txt` or specify desired extras when calling `pip` (e.g. `pip install -v -e .[optional]`). Valid keys for the extras field are: `all`, `tests`, `build`, and `optional`.

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
Note: We set `use_torchvision=True` on-the-fly in CPU mode for `RoIPool` and `RoIAlign`

### Another option: Docker Image

We provide a [Dockerfile](https://github.com/opencv/mmdetection/blob/ote/docker/Dockerfile) to build an image.

```shell
docker build -t otedetection docker/
```

### Prepare datasets

It is recommended to symlink the dataset root to `$OTEDETECTION/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   ├── cityscapes
│   │   ├── annotations
│   │   ├── train
│   │   ├── val
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   ├── VOC2012

```
The cityscapes annotations have to be converted into the coco format using the [cityscapesScripts](https://github.com/mcordts/cityscapesScripts) toolbox.
For the moment we recommend following the instructions provided in the
[maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/tree/master/maskrcnn_benchmark/data) toolbox. When using this script all images have to be moved into the same folder. On linux systems this can e.g. be done for the train images with:
```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmdetection/data mmdetection
```

### A from-scratch setup script

Here is a full script for setting up OTEDetection with conda and link the dataset path (supposing that your COCO dataset path is $COCO_ROOT).

```shell
conda create -n ote-det python=3.7 -y
conda activate ote-det

# install latest pytorch prebuilt with the default prebuilt CUDA version (usually the latest)
conda install -c pytorch pytorch torchvision -y
conda install cython -y
git clone https://github.com/opencv/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .
```

### Using multiple OTEDetection versions

If there are more than one OTEDetection on your machine, and you want to use them alternatively, the recommended way is to create multiple conda environments and use different environments for different versions.

To use the default MMDetection installed in the environment rather than that you are working with, you can remove the following line in those scripts

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```
