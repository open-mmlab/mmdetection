## 依赖

- Linux 和 macOS （Windows 理论上支持）
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ （如果基于 PyTorch 源码安装，也能够支持 CUDA 9.0）
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

MMDetection 和 MMCV 版本兼容性如下所示，需要安装正确的 MMCV 版本以避免安装出现问题。

| MMDetection 版本 |        MMCV 版本         |
| :--------------: | :----------------------: |
|      master      | mmcv-full>=1.3.17, <1.5.0 |
| 2.21.0           | mmcv-full>=1.3.17, <1.5.0 |
| 2.20.0           | mmcv-full>=1.3.17, <1.5.0 |
|      2.19.1      | mmcv-full>=1.3.17, <1.5.0 |
|      2.19.0      | mmcv-full>=1.3.17, <1.5.0 |
|      2.18.1      | mmcv-full>=1.3.17, <1.4.0 |
|      2.18.0      | mmcv-full>=1.3.14, <1.4.0 |
|      2.17.0      | mmcv-full>=1.3.14, <1.4.0 |
|      2.16.0      | mmcv-full>=1.3.8, <1.4.0 |
|      2.15.1      | mmcv-full>=1.3.8, <1.4.0 |
|      2.15.0      | mmcv-full>=1.3.8, <1.4.0 |
|      2.14.0      | mmcv-full>=1.3.8, <1.4.0 |
|      2.13.0      | mmcv-full>=1.3.3, <1.4.0 |
|      2.12.0      | mmcv-full>=1.3.3, <1.4.0 |
|      2.11.0      | mmcv-full>=1.2.4, <1.4.0 |
|      2.10.0      | mmcv-full>=1.2.4, <1.4.0 |
|      2.9.0       | mmcv-full>=1.2.4, <1.4.0 |
|      2.8.0       | mmcv-full>=1.2.4, <1.4.0 |
|      2.7.0       | mmcv-full>=1.1.5, <1.4.0 |
|      2.6.0       | mmcv-full>=1.1.5, <1.4.0 |
|      2.5.0       | mmcv-full>=1.1.5, <1.4.0 |
|      2.4.0       | mmcv-full>=1.1.1, <1.4.0 |
|      2.3.0       |     mmcv-full==1.0.5     |
|     2.3.0rc0     |     mmcv-full>=1.0.2     |
|      2.2.1       |       mmcv==0.6.2        |
|      2.2.0       |       mmcv==0.6.2        |
|      2.1.0       |   mmcv>=0.5.9, <=0.6.1   |
|      2.0.0       |   mmcv>=0.5.1, <=0.5.8   |

**注意：**如果已经安装了 mmcv，首先需要使用 `pip uninstall mmcv` 卸载已安装的 mmcv，如果同时安装了 mmcv 和 mmcv-full，将会报 `ModuleNotFoundError` 错误。

## 安装流程

### 准备环境

1. 使用 conda 新建虚拟环境，并进入该虚拟环境；

   ```shell
   conda create -n open-mmlab python=3.7 -y
   conda activate open-mmlab
   ```

2. 基于 [PyTorch 官网](https://pytorch.org/)安装 PyTorch 和 torchvision，例如：

   ```shell
   conda install pytorch torchvision -c pytorch
   ```

   **注意**：需要确保 CUDA 的编译版本和运行版本匹配。可以在 [PyTorch 官网](https://pytorch.org/)查看预编译包所支持的 CUDA 版本。

   `例 1` 例如在 `/usr/local/cuda` 下安装了 CUDA 10.1， 并想安装 PyTorch 1.5，则需要安装支持 CUDA 10.1 的预构建 PyTorch：

   ```shell
   conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
   ```

   `例 2` 例如在 `/usr/local/cuda` 下安装了 CUDA 9.2， 并想安装 PyTorch 1.3.1，则需要安装支持 CUDA 9.2  的预构建 PyTorch：

   ```shell
   conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
   ```

   如果不是安装预构建的包，而是从源码中构建 PyTorch，则可以使用更多的 CUDA 版本，例如 CUDA 9.0。

### 安装 MMDetection

我们建议使用 [MIM](https://github.com/open-mmlab/mim) 来安装 MMDetection：

``` shell
pip install openmim
mim install mmdet
```

MIM 能够自动地安装 OpenMMLab 的项目以及对应的依赖包。

或者，可以手动安装 MMDetection：

1. 安装 mmcv-full，我们建议使用预构建包来安装：

    ```shell
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
    ```

    需要把命令行中的 `{cu_version}` 和 `{torch_version}` 替换成对应的版本。例如：在 CUDA 11 和 PyTorch 1.7.0 的环境下，可以使用下面命令安装最新版本的 MMCV：

    ```shell
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
    ```

    请参考 [MMCV](https://mmcv.readthedocs.io/en/latest/#installation) 获取不同版本的 MMCV 所兼容的的不同的 PyTorch 和 CUDA 版本。同时，也可以通过以下命令行从源码编译 MMCV：

    ```shell
    git clone https://github.com/open-mmlab/mmcv.git
    cd mmcv
    MMCV_WITH_OPS=1 pip install -e .  # 安装好 mmcv-full
    cd ..
    ```

    或者，可以直接使用命令行安装：

    ```shell
    pip install mmcv-full
    ```

2. 安装 MMDetection：

    你可以直接通过如下命令从 pip 安装使用 mmdetection:

    ```shell
    pip install mmdet
    ```

    或者从 git 仓库编译源码

    ```shell
    git clone https://github.com/open-mmlab/mmdetection.git
    cd mmdetection
    pip install -r requirements/build.txt
    pip install -v -e .  # or "python setup.py develop"
    ```

3. 安装额外的依赖以使用 Instaboost, 全景分割, 或者 LVIS 数据集

    ```shell
    # 安装 instaboost 依赖
    pip install instaboostfast
    # 安装全景分割依赖
    pip install git+https://github.com/cocodataset/panopticapi.git
    # 安装 LVIS 数据集依赖
    pip install git+https://github.com/lvis-dataset/lvis-api.git
    # 安装 albumentations 依赖
    pip install albumentations>=0.3.2 --no-binary imgaug,albumentations
    ```

**注意：**

(1) 按照上述说明，MMDetection 安装在 `dev` 模式下，因此在本地对代码做的任何修改都会生效，无需重新安装；

(2) 如果希望使用 `opencv-python-headless` 而不是 `opencv-python`， 可以在安装 MMCV 之前安装；

(3) 一些安装依赖是可以选择的。例如只需要安装最低运行要求的版本，则可以使用 `pip install -v -e .` 命令。如果希望使用可选择的像 `albumentations` 和 `imagecorruptions` 这种依赖项，可以使用 `pip install -r requirements/optional.txt` 进行手动安装，或者在使用 `pip` 时指定所需的附加功能（例如 `pip install -v -e .[optional]`），支持附加功能的有效键值包括  `all`、`tests`、`build` 以及 `optional` 。

(4) 如果希望使用 `albumentations`，我们建议使用 `pip install albumentations>=0.3.2 --no-binary imgaug,albumentations` 进行安装。 如果简单地使用 `pip install albumentations>=0.3.2` 进行安装，则会同时安装 `opencv-python-headless`（即便已经安装了 `opencv-python` 也会再次安装）。我们不允许同时安装 `opencv-python` 和 `opencv-python-headless`，因为这样可能会导致一些问题。更多细节请参考[官方文档](https://albumentations.ai/docs/getting_started/installation/#note-on-opencv-dependencies)。

### 只在 CPU 安装

我们的代码能够建立在只使用 CPU 的环境（CUDA 不可用）。

在CPU模式下，可以运行 `demo/webcam_demo.py` 示例，然而以下功能将在 CPU 模式下不能使用：

- Deformable Convolution
- Modulated Deformable Convolution
- ROI pooling
- Deformable ROI pooling
- CARAFE: Content-Aware ReAssembly of FEatures
- SyncBatchNorm
- CrissCrossAttention: Criss-Cross Attention
- MaskedConv2d
- Temporal Interlace Shift
- nms_cuda
- sigmoid_focal_loss_cuda
- bbox_overlaps

因此，如果尝试使用包含上述操作的模型进行推理，将会报错。下表列出了由于依赖上述算子而无法在 CPU 上运行的相关模型：

|                        操作                         |                            模型                             |
| :-----------------------------------------------------: | :----------------------------------------------------------: |
| Deformable Convolution/Modulated Deformable Convolution | DCN、Guided Anchoring、RepPoints、CentripetalNet、VFNet、CascadeRPN、NAS-FCOS、DetectoRS |
|                      MaskedConv2d                       |                       Guided Anchoring                       |
|                         CARAFE                          |                            CARAFE                            |
|                      SyncBatchNorm                      |                           ResNeSt                            |

### 另一种选择： Docker 镜像

我们提供了 [Dockerfile](https://github.com/open-mmlab/mmdetection/blob/master/docker/Dockerfile) 来生成镜像，请确保 [docker](https://docs.docker.com/engine/install/)  的版本  >= 19.03。

```shell
# 基于 PyTorch 1.6, CUDA 10.1 生成镜像
docker build -t mmdetection docker/
```

运行命令：

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmdetection/data mmdetection
```

### 从零开始设置脚本

假设当前已经成功安装 CUDA 10.1，这里提供了一个完整的基于 conda 安装 MMDetection 的脚本：

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -y

# 安装最新版本的 mmcv
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html

# 安装 MMDetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .
```

### 使用多个 MMDetection 版本进行开发

训练和测试的脚本已经在 PYTHONPATH 中进行了修改，以确保脚本使用当前目录中的 MMDetection。

要使环境中安装默认的 MMDetection 而不是当前正在在使用的，可以删除出现在相关脚本中的代码：

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```

## 验证

为了验证是否正确安装了 MMDetection 和所需的环境，我们可以运行示例的 Python 代码来初始化检测器并推理一个演示图像：

```python
from mmdet.apis import init_detector, inference_detector

config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
# 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda:0'
# 初始化检测器
model = init_detector(config_file, checkpoint_file, device=device)
# 推理演示图像
inference_detector(model, 'demo/demo.jpg')
```

如果成功安装 MMDetection，则上面的代码可以完整地运行。
