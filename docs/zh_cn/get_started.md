# 开始你的第一步

## 依赖

本节中，我们将演示如何用 PyTorch 准备一个环境。

MMDetection 支持在 Linux，Windows 和 macOS 上运行。它需要 Python 3.7 以上，CUDA 9.2 以上和 PyTorch 1.6 以上。

```{note}
如果你对 PyTorch 有经验并且已经安装了它，你可以直接跳转到[下一小节](#安装流程)。否则，你可以按照下述步骤进行准备。
```

**步骤 0.** 从[官方网站](https://docs.conda.io/en/latest/miniconda.html)下载并安装 Miniconda。

**步骤 1.** 创建并激活一个 conda 环境。

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**步骤 2.** 基于 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)安装 PyTorch。

在 GPU 平台上：

```shell
conda install pytorch torchvision -c pytorch
```

在 CPU 平台上：

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

## 安装流程

我们推荐用户参照我们的最佳实践安装 MMDetection。不过，整个过程也是可定制化的，更多信息请参考[自定义安装](#自定义安装)章节。

### 最佳实践

**步骤 0.** 使用 [MIM](https://github.com/open-mmlab/mim) 安装 [MMEngine](https://github.com/open-mmlab/mmengine) 和 [MMCV](https://github.com/open-mmlab/mmcv)。

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

**注意：** 在 MMCV-v2.x 中，`mmcv-full` 改名为 `mmcv`，如果你想安装不包含 CUDA 算子精简版，可以通过 `mim install "mmcv-lite>=2.0.0rc1"` 来安装。

**步骤 1.** 安装 MMDetection。

方案 a：如果你开发并直接运行 mmdet，从源码安装它：

```shell
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
# "-v" 指详细说明，或更多的输出
# "-e" 表示在可编辑模式下安装项目，因此对代码所做的任何本地修改都会生效，从而无需重新安装。
```

方案 b：如果你将 mmdet 作为依赖或第三方 Python 包，使用 MIM 安装：

```shell
mim install mmdet
```

## 验证安装

为了验证 MMDetection 是否安装正确，我们提供了一些示例代码来执行模型推理。

**步骤 1.** 我们需要下载配置文件和模型权重文件。

```shell
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
```

下载将需要几秒钟或更长时间，这取决于你的网络环境。完成后，你会在当前文件夹中发现两个文件 `rtmdet_tiny_8xb32-300e_coco.py` 和 `rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth`。

**步骤 2.** 推理验证。

方案 a：如果你通过源码安装的 MMDetection，那么直接运行以下命令进行验证：

```shell
python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cpu
```

你会在当前文件夹中的 `outputs/vis` 文件夹中看到一个新的图像 `demo.jpg`，图像中包含有网络预测的检测框。

方案 b：如果你通过 MIM 安装的 MMDetection，那么可以打开你的 Python 解析器，复制并粘贴以下代码：

```python
from mmdet.apis import init_detector, inference_detector

config_file = 'rtmdet_tiny_8xb32-300e_coco.py'
checkpoint_file = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
inference_detector(model, 'demo/demo.jpg')
```

你将会看到一个包含 `DetDataSample` 的列表，预测结果在 `pred_instance` 里，包含有检测框，类别和得分。

### 自定义安装

#### CUDA 版本

在安装 PyTorch 时，你需要指定 CUDA 的版本。如果你不清楚应该选择哪一个，请遵循我们的建议：

- 对于 Ampere 架构的 NVIDIA GPU，例如 GeForce 30 系列以及 NVIDIA A100，CUDA 11 是必需的。
- 对于更早的 NVIDIA GPU，CUDA 11 是向后兼容 (backward compatible) 的，但 CUDA 10.2 能够提供更好的兼容性，也更加轻量。

请确保你的 GPU 驱动版本满足最低的版本需求，参阅 NVIDIA 官方的 [CUDA 工具箱和相应的驱动版本关系表](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions)。

```{note}
如果按照我们的最佳实践，安装 CUDA 运行时库就足够了，这是因为不需要在本地编译 CUDA 代码。但如果你希望从源码编译 MMCV，或是开发其他 CUDA 算子，那么就必须安装完整的 CUDA 工具链，参见 [NVIDIA 官网](https://developer.nvidia.com/cuda-downloads)，另外还需要确保该 CUDA 工具链的版本与 PyTorch 安装时的配置相匹配（如用 `conda install` 安装 PyTorch 时指定的 cudatoolkit 版本）。
```

#### 不使用 MIM 安装 MMEngine

要使用 pip 而不是 MIM 来安装 MMEngine，请遵照 [MMEngine 安装指南](https://mmengine.readthedocs.io/zh_CN/latest/get_started/installation.html)。

例如，你可以通过以下命令安装 MMEngine。

```shell
pip install mmengine
```

#### 不使用 MIM 安装 MMCV

MMCV 包含 C++ 和 CUDA 扩展，因此其对 PyTorch 的依赖比较复杂。MIM 会自动解析这些依赖，选择合适的 MMCV 预编译包，使安装更简单，但它并不是必需的。

要使用 pip 而不是 MIM 来安装 MMCV，请遵照 [MMCV 安装指南](https://mmcv.readthedocs.io/zh_CN/2.x/get_started/installation.html)。它需要您用指定 url 的形式手动指定对应的 PyTorch 和 CUDA 版本。

例如，下述命令将会安装基于 PyTorch 1.12.x 和 CUDA 11.6 编译的 MMCV。

```shell
pip install "mmcv>=2.0.0" -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html
```

#### 在 CPU 环境中安装

MMDetection 可以在 CPU 环境中构建。在 CPU 模式下，可以进行模型训练（需要 MMCV 版本 >= 2.0.0rc1）、测试或者推理。

但是，以下功能在该模式下不能使用：

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

因此，如果尝试训练/测试/推理包含上述算子的模型，将会报错。下表列出了将会受影响的相关算法。

|                          操作                           |                                           模型                                           |
| :-----------------------------------------------------: | :--------------------------------------------------------------------------------------: |
| Deformable Convolution/Modulated Deformable Convolution | DCN、Guided Anchoring、RepPoints、CentripetalNet、VFNet、CascadeRPN、NAS-FCOS、DetectoRS |
|                      MaskedConv2d                       |                                     Guided Anchoring                                     |
|                         CARAFE                          |                                          CARAFE                                          |
|                      SyncBatchNorm                      |                                         ResNeSt                                          |

#### 在 Google Colab 中安装

[Google Colab](https://colab.research.google.com/) 通常已经包含了 PyTorch 环境，因此我们只需要安装 MMEngine，MMCV 和 MMDetection 即可，命令如下：

**步骤 1.** 使用 [MIM](https://github.com/open-mmlab/mim) 安装 [MMEngine](https://github.com/open-mmlab/mmengine) 和 [MMCV](https://github.com/open-mmlab/mmcv)。

```shell
!pip3 install openmim
!mim install mmengine
!mim install "mmcv>=2.0.0,<2.1.0"
```

**步骤 2.** 使用源码安装 MMDetection。

```shell
!git clone https://github.com/open-mmlab/mmdetection.git
%cd mmdetection
!pip install -e .
```

**步骤 3.** 验证安装是否成功。

```python
import mmdet
print(mmdet.__version__)
# 预期输出：3.0.0 或其他版本号
```

```{note}
在 Jupyter Notebook 中，感叹号 `!` 用于执行外部命令，而 `%cd` 是一个[魔术命令](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-cd)，用于切换 Python 的工作路径。
```

#### 通过 Docker 使用 MMDetection

我们提供了一个 [Dockerfile](../../docker/Dockerfile) 来构建一个镜像。请确保你的 [docker 版本](https://docs.docker.com/engine/install/) >=19.03。

```shell
# 基于 PyTorch 1.9，CUDA 11.1 构建镜像
# 如果你想要其他版本，只需要修改 Dockerfile
docker build -t mmdetection docker/
```

用以下命令运行 Docker 镜像：

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmdetection/data mmdetection
```

### 排除故障

如果你在安装过程中遇到一些问题，请先查看 [FAQ](notes/faq.md) 页面。如果没有找到解决方案，你也可以在 GitHub 上[提出一个问题](https://github.com/open-mmlab/mmdetection/issues/new/choose)。

### 使用多个 MMDetection 版本进行开发

训练和测试的脚本已经在 `PYTHONPATH` 中进行了修改，以确保脚本使用当前目录中的 MMDetection。

要使环境中安装默认版本的 MMDetection 而不是当前正在使用的，可以删除出现在相关脚本中的代码：

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```
