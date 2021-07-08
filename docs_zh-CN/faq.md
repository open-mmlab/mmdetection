# 常见问题解答

我们在这里列出了使用时的一些常见问题及其相应的解决方案。 如果您发现有一些问题被遗漏，请随时提 PR 丰富这个列表。 如果您无法在此获得帮助，请使用 [issue模板](https://github.com/open-mmlab/mmdetection/blob/master/.github/ISSUE_TEMPLATE/error-report.md/ )创建问题，但是请在模板中填写所有必填信息，这有助于我们更快定位问题。

## MMCV 安装相关

- MMCV 与 MMDetection 的兼容问题: "ConvWS is already registered in conv layer"; "AssertionError: MMCV==xxx is used but incompatible. Please install mmcv>=xxx, <=xxx."

  请按 [安装说明](https://mmdetection.readthedocs.io/zh_CN/latest/get_started.html#installation) 为你的 MMDetection 安装正确版本的 MMCV 。

- "No module named 'mmcv.ops'"; "No module named 'mmcv._ext'".

    原因是安装了 `mmcv` 而不是 `mmcv-full`。

    1. `pip uninstall mmcv` 卸载安装的 `mmcv`

    2. 安装 `mmcv-full` 根据 [安装说明](https://mmcv.readthedocs.io/zh/latest/#installation)。

## PyTorch/CUDA 环境相关

- "RTX 30 series card fails when building MMCV or MMDet"

    1. 临时解决方案为使用命令 `MMCV_WITH_OPS=1 MMCV_CUDA_ARGS='-gencode=arch=compute_80,code=sm_80' pip install -e .` 进行编译。 常见报错信息为 `nvcc fatal : Unsupported gpu architecture 'compute_86'` 意思是你的编译器不支持 sm_86 架构(包括英伟达 30 系列的显卡)的优化，至 CUDA toolkit 11.0 依旧未支持. 这个命令是通过增加宏 `MMCV_CUDA_ARGS='-gencode=arch=compute_80,code=sm_80` 让 nvcc 编译器为英伟达 30 系列显卡进行 `sm_80` 的优化，虽然这有可能会无法发挥出显卡所有性能。

    2. 有开发者已经在 [pytorch/pytorch#47585](https://github.com/pytorch/pytorch/pull/47585) 更新了 PyTorch 默认的编译 flag， 但是我们对此并没有进行测试。

- "invalid device function" or "no kernel image is available for execution".

    1. 检查您正常安装了 CUDA runtime (一般在`/usr/local/`)，或者使用 `nvcc --version` 检查本地版本，有时安装 PyTorch 会顺带安装一个 CUDA runtime，并且实际优先使用 conda 环境中的版本，你可以使用 `conda list cudatoolkit` 查看其版本。

    2. 编译 extention 的 CUDA Toolkit 版本与运行时的 CUDA Toolkit 版本是否相符，

       * 如果您从源码自己编译的，使用 `python mmdet/utils/collect_env.py` 检查编译编译 extention 的 CUDA Toolkit 版本，然后使用 `conda list cudatoolkit` 检查当前 conda 环境是否有 CUDA Toolkit，若有检查版本是否匹配， 如不匹配，更换 conda 环境的 CUDA Toolkit，或者使用匹配的 CUDA Toolkit 中的 nvcc 编译即可，如环境中无 CUDA Toolkit，可以使用 `nvcc -V`。

         等命令查看当前使用的 CUDA runtime。

       * 如果您是通过 pip 下载的预编译好的版本，请确保与当前 CUDA runtime 一致。

    3. 运行 `python mmdet/utils/collect_env.py` 检查是否为正确的 GPU 架构编译的 PyTorch， torchvision， 与 MMCV。 你或许需要设置 `TORCH_CUDA_ARCH_LIST` 来重新安装 MMCV，可以参考 [GPU 架构表](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list),
        例如， 运行 `TORCH_CUDA_ARCH_LIST=7.0 pip install mmcv-full` 为 Volta GPU 编译 MMCV。这种架构不匹配的问题一般会出现在使用一些旧型号的 GPU 时候出现， 例如， Tesla K80。

- "undefined symbol" or "cannot open xxx.so".

    1. 如果这些 symbol 属于 CUDA/C++ (如 libcudart.so 或者 GLIBCXX)，使用 `python mmdet/utils/collect_env.py`检查 CUDA/GCC runtime 与编译 MMCV 的 CUDA 版本是否相同。
    2. 如果这些 symbols 属于 PyTorch，(例如, symbols containing caffe, aten, and TH), 检查当前 Pytorch 版本是否与编译 MMCV 的版本一致。
    3. 运行 `python mmdet/utils/collect_env.py` 检查 PyTorch， torchvision， MMCV 等的编译环境与运行环境一致。

- setuptools.sandbox.UnpickleableException: DistutilsSetupError("each element of 'ext_modules' option must be an Extension instance or 2-tuple")

    1. 如果你在使用 miniconda 而不是 anaconda，检查是否正确的安装了 Cython 如 [#3379](https://github.com/open-mmlab/mmdetection/issues/3379).
    2. 检查环境中的 `setuptools`, `Cython`, and `PyTorch` 相互之间版本是否匹配。

- "Segmentation fault".
    1. 检查 GCC 的版本，通常是因为 PyTorch 版本与 GCC 版本不匹配 （例如 GCC < 4.9 )，我们推荐用户使用 GCC 5.4，我们也不推荐使用 GCC 5.5， 因为有反馈 GCC 5.5 会导致 "segmentation fault" 并且切换到 GCC 5.4 就可以解决问题。

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
- ’GPU out of memory"
    1. 存在大量 ground truth boxes 或者大量 anchor 的场景，可能在 assigner 会 OOM。 您可以在 assigner 的配置中设置 `gpu_assign_thr=N`，这样当超过 N 个 GT boxes 时，assigner 会通过 CPU 计算 IOU。
    2. 在 backbone 中设置 `with_cp=True`。 这使用 PyTorch 中的 `sublinear strategy` 来降低 backbone 占用的 GPU 显存。
    3. 使用 `config/fp16` 中的示例尝试混合精度训练。`loss_scale` 可能需要针对不同模型进行调整。
- "RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one"
    1. 这个错误出现在存在参数没有在 forward 中使用，容易在 DDP 中运行不同分支时发生。
    2. 你可以在 config 设置 `find_unused_parameters = True`，或者手动查找哪些参数没有用到。

## Evaluation 相关

- 使用 COCO Dataset 的测评接口时, 测评结果中 AP 或者 AR = -1
    1. 根据COCO数据集的定义，一张图像中的中等物体与小物体面积的阈值分别为 9216（96\*96）与 1024（32\*32）。
    2. 如果在某个区间没有检测框 AP 与 AR 认定为 -1.
