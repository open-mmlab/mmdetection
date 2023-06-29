# 常见问题解答

我们在这里列出了使用时的一些常见问题及其相应的解决方案。 如果您发现有一些问题被遗漏，请随时提 PR 丰富这个列表。 如果您无法在此获得帮助，请使用 [issue模板](https://github.com/open-mmlab/mmdetection/blob/main/.github/ISSUE_TEMPLATE/error-report.md/)创建问题，但是请在模板中填写所有必填信息，这有助于我们更快定位问题。

## PyTorch 2.0 支持

MMDetection 目前绝大部分算法已经支持了 PyTorch 2.0 及其 `torch.compile` 功能, 用户只需要安装 MMDetection 3.0.0rc7 及其以上版本即可。如果你在使用中发现有不支持的算法，欢迎给我们反馈。我们也非常欢迎社区贡献者来 benchmark 对比 `torch.compile` 功能所带来的速度提升。

如果你想启动 `torch.compile` 功能，只需要在 `train.py` 或者 `test.py` 后面加上 `--cfg-options compile=True`。 以 RTMDet 为例，你可以使用以下命令启动 `torch.compile` 功能：

```shell
# 单卡
python tools/train.py configs/rtmdet/rtmdet_s_8xb32-300e_coco.py  --cfg-options compile=True

# 单机 8 卡
./tools/dist_train.sh configs/rtmdet/rtmdet_s_8xb32-300e_coco.py 8 --cfg-options compile=True

# 单机 8 卡 + AMP 混合精度训练
./tools/dist_train.sh configs/rtmdet/rtmdet_s_8xb32-300e_coco.py 8 --cfg-options compile=True --amp
```

需要特别注意的是，PyTorch 2.0 对于动态 shape 支持不是非常完善，目标检测算法中大部分不仅输入 shape 是动态的，而且 loss 计算和后处理过程中也是动态的，这会导致在开启 `torch.compile` 功能后训练速度会变慢。基于此，如果你想启动 `torch.compile` 功能，则应该遵循如下原则：

1. 输入到网络的图片是固定 shape 的，而非多尺度的
2. 设置 `torch._dynamo.config.cache_size_limit` 参数。TorchDynamo 会将 Python 字节码转换并缓存，已编译的函数会被存入缓存中。当下一次检查发现需要重新编译时，该函数会被重新编译并缓存。但是如果重编译次数超过预设的最大值（64），则该函数将不再被缓存或重新编译。前面说过目标检测算法中的 loss 计算和后处理部分也是动态计算的，这些函数需要在每次迭代中重新编译。因此将 `torch._dynamo.config.cache_size_limit` 参数设置得更小一些可以有效减少编译时间

在 MMDetection 中可以通过环境变量 `DYNAMO_CACHE_SIZE_LIMIT` 设置 `torch._dynamo.config.cache_size_limit` 参数，以 RTMDet 为例，命令如下所示：

```shell
# 单卡
export DYNAMO_CACHE_SIZE_LIMIT = 4
python tools/train.py configs/rtmdet/rtmdet_s_8xb32-300e_coco.py  --cfg-options compile=True

# 单机 8 卡
export DYNAMO_CACHE_SIZE_LIMIT = 4
./tools/dist_train.sh configs/rtmdet/rtmdet_s_8xb32-300e_coco.py 8 --cfg-options compile=True
```

关于 PyTorch 2.0 的 dynamo 常见问题，可以参考 [这里](https://pytorch.org/docs/stable/dynamo/faq.html)

## 安装

- MMCV 与 MMDetection 的兼容问题: "ConvWS is already registered in conv layer"; "AssertionError: MMCV==xxx is used but incompatible. Please install mmcv>=xxx, \<=xxx."

  MMDetection，MMEngine 和 MMCV 的版本兼容关系如下。请选择合适的版本避免安装错误 。

  | MMDetection 版本 |        MMCV 版本        |      MMEngine 版本       |
  | :--------------: | :---------------------: | :----------------------: |
  |       main       |  mmcv>=2.0.0, \<2.1.0   | mmengine>=0.7.1, \<1.0.0 |
  |      3.1.0       |  mmcv>=2.0.0, \<2.1.0   | mmengine>=0.7.1, \<1.0.0 |
  |      3.0.0       |  mmcv>=2.0.0, \<2.1.0   | mmengine>=0.7.1, \<1.0.0 |
  |     3.0.0rc6     | mmcv>=2.0.0rc4, \<2.1.0 | mmengine>=0.6.0, \<1.0.0 |
  |     3.0.0rc5     | mmcv>=2.0.0rc1, \<2.1.0 | mmengine>=0.3.0, \<1.0.0 |
  |     3.0.0rc4     | mmcv>=2.0.0rc1, \<2.1.0 | mmengine>=0.3.0, \<1.0.0 |
  |     3.0.0rc3     | mmcv>=2.0.0rc1, \<2.1.0 | mmengine>=0.3.0, \<1.0.0 |
  |     3.0.0rc2     | mmcv>=2.0.0rc1, \<2.1.0 | mmengine>=0.1.0, \<1.0.0 |
  |     3.0.0rc1     | mmcv>=2.0.0rc1, \<2.1.0 | mmengine>=0.1.0, \<1.0.0 |
  |     3.0.0rc0     | mmcv>=2.0.0rc1, \<2.1.0 | mmengine>=0.1.0, \<1.0.0 |

  **注意：**

  1. 如果你希望安装 mmdet-v2.x, MMDetection 和 MMCV 版本兼容表可以在 [这里](https://mmdetection.readthedocs.io/en/stable/faq.html#installation) 找到，请选择合适的版本避免安装错误。
  2. 在 MMCV-v2.x 中，`mmcv-full` 改名为 `mmcv`，如果你想安装不包含 CUDA 算子的版本，可以选择安装 MMCV 精简版 `mmcv-lite`。

- "No module named 'mmcv.ops'"; "No module named 'mmcv.\_ext'".

  原因是安装了 `mmcv-lite` 而不是 `mmcv`。

  1. `pip uninstall mmcv-lite` 卸载安装的 `mmcv-lite`

  2. 安装 `mmcv` 根据 [安装说明](https://mmcv.readthedocs.io/zh_CN/2.x/get_started/installation.html)。

- 在 Windows 环境下安装过程中遇到 "Microsoft Visual C++ 14.0 or graeter is required" error .

  这个错误发生在 pycotools 的 'pycocotools.\_mask' 扩展构建过程，其原因是缺少了对应 C++ 环境依赖。你需要到微软官方下载[对应工具](https://visualstudio.microsoft.com/zh-hans/visual-cpp-build-tools/)，选择“使用 C++ 的桌面开发”选项安装最小依赖，随后重新安装 pycocotools。

- 使用 albumentations

如果你希望使用 `albumentations`，我们建议使用 `pip install -r requirements/albu.txt`
或者 `pip install -U albumentations --no-binary qudida,albumentations` 进行安装。
如果简单地使用 `pip install albumentations>=0.3.2` 进行安装，
则会同时安装 `opencv-python-headless`（即便已经安装了 `opencv-python` 也会再次安装）。
我们建议在安装 `albumentations` 后检查环境，以确保没有同时安装 `opencv-python` 和 `opencv-python-headless`，
因为同时安装可能会导致一些问题。更多细节请参考[官方文档](https://albumentations.ai/docs/getting_started/installation/#note-on-opencv-dependencies) 。

- 在某些算法中出现 ModuleNotFoundError 错误

一些算法或者数据需要额外的依赖，例如 Instaboost、 Panoptic Segmentation、 LVIS dataset 等。请注意错误信息并安装相应的包，例如：

```shell
# 安装 instaboost 依赖
pip install instaboostfast
# 安装 panoptic segmentation 依赖
pip install git+https://github.com/cocodataset/panopticapi.git
# 安装 LVIS dataset 依赖
pip install git+https://github.com/lvis-dataset/lvis-api.git
```

## 代码

- 修改一些代码后是否需要重新安装 mmdet

如果你遵循最佳实践，即使用 `pip install -v -e .` 安装的 mmdet，则对本地代码所作的任何修改都会生效，无需重新安装

- 如何使用多个 MMDetection 版本进行开发

你可以拥有多个文件夹，例如 mmdet-3.0，mmdet-3.1。

要使环境中安装默认的 MMDetection 而不是当前正在在使用的，可以删除出现在相关脚本中的代码：

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```

## PyTorch/CUDA 环境相关

- "RTX 30 series card fails when building MMCV or MMDet"

  1. 临时解决方案为使用命令 `MMCV_WITH_OPS=1 MMCV_CUDA_ARGS='-gencode=arch=compute_80,code=sm_80' pip install -e .` 进行编译。 常见报错信息为 `nvcc fatal : Unsupported gpu architecture 'compute_86'` 意思是你的编译器不支持 sm_86 架构(包括英伟达 30 系列的显卡)的优化，至 CUDA toolkit 11.0 依旧未支持. 这个命令是通过增加宏 `MMCV_CUDA_ARGS='-gencode=arch=compute_80,code=sm_80` 让 nvcc 编译器为英伟达 30 系列显卡进行 `sm_80` 的优化，虽然这有可能会无法发挥出显卡所有性能。

  2. 有开发者已经在 [pytorch/pytorch#47585](https://github.com/pytorch/pytorch/pull/47585) 更新了 PyTorch 默认的编译 flag， 但是我们对此并没有进行测试。

- "invalid device function" 或者 "no kernel image is available for execution".

  1. 检查您正常安装了 CUDA runtime (一般在`/usr/local/`)，或者使用 `nvcc --version` 检查本地版本，有时安装 PyTorch 会顺带安装一个 CUDA runtime，并且实际优先使用 conda 环境中的版本，你可以使用 `conda list cudatoolkit` 查看其版本。

  2. 编译 extension 的 CUDA Toolkit 版本与运行时的 CUDA Toolkit 版本是否相符，

     - 如果您从源码自己编译的，使用 `python mmdet/utils/collect_env.py` 检查编译编译 extension 的 CUDA Toolkit 版本，然后使用 `conda list cudatoolkit` 检查当前 conda 环境是否有 CUDA Toolkit，若有检查版本是否匹配， 如不匹配，更换 conda 环境的 CUDA Toolkit，或者使用匹配的 CUDA Toolkit 中的 nvcc 编译即可，如环境中无 CUDA Toolkit，可以使用 `nvcc -V`。

       等命令查看当前使用的 CUDA runtime。

     - 如果您是通过 pip 下载的预编译好的版本，请确保与当前 CUDA runtime 一致。

  3. 运行 `python mmdet/utils/collect_env.py` 检查是否为正确的 GPU 架构编译的 PyTorch， torchvision， 与 MMCV。 你或许需要设置 `TORCH_CUDA_ARCH_LIST` 来重新安装 MMCV，可以参考 [GPU 架构表](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list),
     例如， 运行 `TORCH_CUDA_ARCH_LIST=7.0 pip install mmcv` 为 Volta GPU 编译 MMCV。这种架构不匹配的问题一般会出现在使用一些旧型号的 GPU 时候出现， 例如， Tesla K80。

- "undefined symbol" 或者 "cannot open xxx.so".

  1. 如果这些 symbol 属于 CUDA/C++ (如 libcudart.so 或者 GLIBCXX)，使用 `python mmdet/utils/collect_env.py`检查 CUDA/GCC runtime 与编译 MMCV 的 CUDA 版本是否相同。
  2. 如果这些 symbols 属于 PyTorch，(例如, symbols containing caffe, aten, and TH), 检查当前 Pytorch 版本是否与编译 MMCV 的版本一致。
  3. 运行 `python mmdet/utils/collect_env.py` 检查 PyTorch， torchvision， MMCV 等的编译环境与运行环境一致。

- setuptools.sandbox.UnpickleableException: DistutilsSetupError("each element of 'ext_modules' option must be an Extension instance or 2-tuple")

  1. 如果你在使用 miniconda 而不是 anaconda，检查是否正确的安装了 Cython 如 [#3379](https://github.com/open-mmlab/mmdetection/issues/3379).
  2. 检查环境中的 `setuptools`, `Cython`, and `PyTorch` 相互之间版本是否匹配。

- "Segmentation fault".

  1. 检查 GCC 的版本，通常是因为 PyTorch 版本与 GCC 版本不匹配 （例如 GCC \< 4.9 )，我们推荐用户使用 GCC 5.4，我们也不推荐使用 GCC 5.5， 因为有反馈 GCC 5.5 会导致 "segmentation fault" 并且切换到 GCC 5.4 就可以解决问题。

  2. 检查是否正确安装了 CUDA 版本的 PyTorch 。

     ```shell
     python -c 'import torch; print(torch.cuda.is_available())'
     ```

     是否返回True。

  3. 如果 `torch` 的安装是正确的，检查是否正确编译了 MMCV。

     ```shell
     python -c 'import mmcv; import mmcv.ops'
     ```

  4. 如果 MMCV 与 PyTorch 都被正确安装了，则使用 `ipdb`, `pdb` 设置断点，直接查找哪一部分的代码导致了 `segmentation fault`。

## Training 相关

- "Loss goes Nan"

  1. 检查数据的标注是否正常， 长或宽为 0 的框可能会导致回归 loss 变为 nan，一些小尺寸（宽度或高度小于 1）的框在数据增强（例如，instaboost）后也会导致此问题。 因此，可以检查标注并过滤掉那些特别小甚至面积为 0 的框，并关闭一些可能会导致 0 面积框出现数据增强。
  2. 降低学习率：由于某些原因，例如 batch size 大小的变化， 导致当前学习率可能太大。 您可以降低为可以稳定训练模型的值。
  3. 延长 warm up 的时间：一些模型在训练初始时对学习率很敏感，您可以把 `warmup_iters` 从 500 更改为 1000 或 2000。
  4. 添加 gradient clipping: 一些模型需要梯度裁剪来稳定训练过程。 默认的 `grad_clip` 是 `None`,  你可以在 config 设置 `optimizer_config=dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))`  如果你的 config 没有继承任何包含 `optimizer_config=dict(grad_clip=None)`,  你可以直接设置`optimizer_config=dict(grad_clip=dict(max_norm=35, norm_type=2))`.

- "GPU out of memory"

  1. 存在大量 ground truth boxes 或者大量 anchor 的场景，可能在 assigner 会 OOM。 您可以在 assigner 的配置中设置 `gpu_assign_thr=N`，这样当超过 N 个 GT boxes 时，assigner 会通过 CPU 计算 IOU。

  2. 在 backbone 中设置 `with_cp=True`。 这使用 PyTorch 中的 `sublinear strategy` 来降低 backbone 占用的 GPU 显存。

  3. 使用 `config/fp16` 中的示例尝试混合精度训练。`loss_scale` 可能需要针对不同模型进行调整。

  4. 你也可以尝试使用 `AvoidCUDAOOM` 来避免该问题。首先它将尝试调用 `torch.cuda.empty_cache()`。如果失败，将会尝试把输入类型转换到 FP16。如果仍然失败，将会把输入从 GPUs 转换到 CPUs 进行计算。这里提供了两个使用的例子：

     ```python
     from mmdet.utils import AvoidCUDAOOM

     output = AvoidCUDAOOM.retry_if_cuda_oom(some_function)(input1, input2)
     ```

     你也可也使用 `AvoidCUDAOOM` 作为装饰器让代码遇到 OOM 的时候继续运行：

     ```python
     from mmdet.utils import AvoidCUDAOOM

     @AvoidCUDAOOM.retry_if_cuda_oom
     def function(*args, **kwargs):
         ...
         return xxx
     ```

- "RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one"

  1. 这个错误出现在存在参数没有在 forward 中使用，容易在 DDP 中运行不同分支时发生。
  2. 你可以在 config 设置 `find_unused_parameters = True` 进行训练 (会降低训练速度)。
  3. 你也可以通过在 config 中的 `optimizer_config` 里设置 `detect_anomalous_params=True` 查找哪些参数没有用到，但是需要 MMCV 的版本 >= 1.4.1。

- 训练中保存最好模型

  可以通过配置 `default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto')`开启。在 `auto` 参数情况下会根据返回的验证结果中的第一个 key 作为选择最优模型的依据，你也可以直接设置评估结果中的 key 来手动设置，例如 `save_best='coco/bbox_mAP'`。

- 在 Resume 训练中使用 `ExpMomentumEMAHook`

  如果在训练中使用了 `ExpMomentumEMAHook`，那么 resume 时候不能仅仅通过命令行参数 `--resume-from` 或 `--cfg-options resume_from` 实现恢复模型参数功能例如 `python tools/train.py configs/yolox/yolox_s_8x8_300e_coco.py --resume-from ./work_dir/yolox_s_8x8_300e_coco/epoch_x.pth`。以 `yolox_s` 算法为例，由于 `ExpMomentumEMAHook` 需要重新加载权重，你可以通过如下做法实现：

  ```python
  # 直接打开 configs/yolox/yolox_s_8x8_300e_coco.py 修改所有 resume_from 字段
  resume_from=./work_dir/yolox_s_8x8_300e_coco/epoch_x.pth
  custom_hooks=[...
      dict(
          type='ExpMomentumEMAHook',
          resume_from=./work_dir/yolox_s_8x8_300e_coco/epoch_x.pth,
          momentum=0.0001,
          priority=49)
      ]
  ```

## Evaluation 相关

- 使用 COCO Dataset 的测评接口时, 测评结果中 AP 或者 AR = -1
  1. 根据COCO数据集的定义，一张图像中的中等物体与小物体面积的阈值分别为 9216（96\*96）与 1024（32\*32）。
  2. 如果在某个区间没有检测框 AP 与 AR 认定为 -1.

## Model 相关

- **ResNet style 参数说明**

  ResNet style 可选参数允许 `pytorch` 和 `caffe`，其差别在于 Bottleneck 模块。Bottleneck 是 `1x1-3x3-1x1` 堆叠结构，在 `caffe` 模式模式下 stride=2 参数放置在第一个 `1x1` 卷积处，而 `pyorch` 模式下 stride=2 放在第二个 `3x3` 卷积处。一个简单示例如下：

  ```python
  if self.style == 'pytorch':
        self.conv1_stride = 1
        self.conv2_stride = stride
  else:
        self.conv1_stride = stride
        self.conv2_stride = 1
  ```

- **ResNeXt 参数说明**

  ResNeXt 来自论文 [`Aggregated Residual Transformations for Deep Neural Networks`](https://arxiv.org/abs/1611.05431). 其引入分组卷积，并且通过变量基数来控制组的数量达到精度和复杂度的平衡，其有两个超参 `baseWidth` 和 `cardinality `来控制内部 Bottleneck 模块的基本宽度和分组数参数。以 MMDetection 中配置名为 `mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco.py` 为例，其中 `mask_rcnn` 代表算法采用 Mask R-CNN，`x101` 代表骨架网络采用 ResNeXt-101，`64x4d`代表 Bottleneck 一共分成 64 组，每组的基本宽度是 4。

- **骨架网络 eval 模式说明**

  因为检测模型通常比较大且输入图片分辨率很高，这会导致检测模型的 batch 很小，通常是 2，这会使得 BatchNorm 在训练过程计算的统计量方差非常大，不如主干网络预训练时得到的统计量稳定，因此在训练是一般都会使用 `norm_eval=True` 模式，直接使用预训练主干网络中的 BatchNorm 统计量，少数使用大 batch 的算法是 `norm_eval=False` 模式，例如 NASFPN。对于没有 ImageNet 预训练的骨架网络，如果 batch 比较小，可以考虑使用 `SyncBN`。
