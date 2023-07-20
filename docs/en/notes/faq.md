# Frequently Asked Questions

We list some common troubles faced by many users and their corresponding solutions here. Feel free to enrich the list if you find any frequent issues and have ways to help others to solve them. If the contents here do not cover your issue, please create an issue using the [provided templates](https://github.com/open-mmlab/mmdetection/blob/main/.github/ISSUE_TEMPLATE/error-report.md/) and make sure you fill in all required information in the template.

## PyTorch 2.0 Support

The vast majority of algorithms in MMDetection now support PyTorch 2.0 and its `torch.compile` function. Users only need to install MMDetection 3.0.0rc7 or later versions to enjoy this feature. If any unsupported algorithms are found during use, please feel free to give us feedback. We also welcome contributions from the community to benchmark the speed improvement brought by using the `torch.compile` function.

To enable the `torch.compile` function, simply add `--cfg-options compile=True` after `train.py` or `test.py`. For example, to enable `torch.compile` for RTMDet, you can use the following command:

```shell
# Single GPU
python tools/train.py configs/rtmdet/rtmdet_s_8xb32-300e_coco.py  --cfg-options compile=True

# Single node multiple GPUs
./tools/dist_train.sh configs/rtmdet/rtmdet_s_8xb32-300e_coco.py 8 --cfg-options compile=True

# Single node multiple GPUs + AMP
./tools/dist_train.sh configs/rtmdet/rtmdet_s_8xb32-300e_coco.py 8 --cfg-options compile=True --amp
```

It is important to note that PyTorch 2.0's support for dynamic shapes is not yet fully developed. In most object detection algorithms, not only are the input shapes dynamic, but the loss calculation and post-processing parts are also dynamic. This can lead to slower training speeds when using the `torch.compile` function. Therefore, if you wish to enable the `torch.compile` function, you should follow these principles:

1. Input images to the network are fixed shape, not multi-scale
2. set `torch._dynamo.config.cache_size_limit` parameter. TorchDynamo will convert and cache the Python bytecode, and the compiled functions will be stored in the cache. When the next check finds that the function needs to be recompiled, the function will be recompiled and cached. However, if the number of recompilations exceeds the maximum value set (64), the function will no longer be cached or recompiled. As mentioned above, the loss calculation and post-processing parts of the object detection algorithm are also dynamically calculated, and these functions need to be recompiled every time. Therefore, setting the `torch._dynamo.config.cache_size_limit` parameter to a smaller value can effectively reduce the compilation time

In MMDetection, you can set the `torch._dynamo.config.cache_size_limit` parameter through the environment variable `DYNAMO_CACHE_SIZE_LIMIT`. For example, the command is as follows:

```shell
# Single GPU
export DYNAMO_CACHE_SIZE_LIMIT = 4
python tools/train.py configs/rtmdet/rtmdet_s_8xb32-300e_coco.py  --cfg-options compile=True

# Single node multiple GPUs
export DYNAMO_CACHE_SIZE_LIMIT = 4
./tools/dist_train.sh configs/rtmdet/rtmdet_s_8xb32-300e_coco.py 8 --cfg-options compile=True
```

About the common questions about PyTorch 2.0's dynamo, you can refer to [here](https://pytorch.org/docs/stable/dynamo/faq.html)

## Installation

Compatibility issue between MMCV and MMDetection; "ConvWS is already registered in conv layer"; "AssertionError: MMCV==xxx is used but incompatible. Please install mmcv>=xxx, \<=xxx."

Compatible MMDetection, MMEngine, and MMCV versions are shown as below. Please choose the correct version of MMCV to avoid installation issues.

| MMDetection version |      MMCV version       |     MMEngine version     |
| :-----------------: | :---------------------: | :----------------------: |
|        main         |  mmcv>=2.0.0, \<2.1.0   | mmengine>=0.7.1, \<1.0.0 |
|        3.1.0        |  mmcv>=2.0.0, \<2.1.0   | mmengine>=0.7.1, \<1.0.0 |
|        3.0.0        |  mmcv>=2.0.0, \<2.1.0   | mmengine>=0.7.1, \<1.0.0 |
|      3.0.0rc6       | mmcv>=2.0.0rc4, \<2.1.0 | mmengine>=0.6.0, \<1.0.0 |
|      3.0.0rc5       | mmcv>=2.0.0rc1, \<2.1.0 | mmengine>=0.3.0, \<1.0.0 |
|      3.0.0rc4       | mmcv>=2.0.0rc1, \<2.1.0 | mmengine>=0.3.0, \<1.0.0 |
|      3.0.0rc3       | mmcv>=2.0.0rc1, \<2.1.0 | mmengine>=0.3.0, \<1.0.0 |
|      3.0.0rc2       | mmcv>=2.0.0rc1, \<2.1.0 | mmengine>=0.1.0, \<1.0.0 |
|      3.0.0rc1       | mmcv>=2.0.0rc1, \<2.1.0 | mmengine>=0.1.0, \<1.0.0 |
|      3.0.0rc0       | mmcv>=2.0.0rc1, \<2.1.0 | mmengine>=0.1.0, \<1.0.0 |

**Note:**

1. If you want to install mmdet-v2.x, the compatible MMDetection and MMCV versions table can be found at [here](https://mmdetection.readthedocs.io/en/stable/faq.html#installation). Please choose the correct version of MMCV to avoid installation issues.
2. In MMCV-v2.x, `mmcv-full` is rename to `mmcv`, if you want to install `mmcv` without CUDA ops, you can install `mmcv-lite`.

- "No module named 'mmcv.ops'"; "No module named 'mmcv.\_ext'".

  1. Uninstall existing `mmcv-lite` in the environment using `pip uninstall mmcv-lite`.
  2. Install `mmcv` following the [installation instruction](https://mmcv.readthedocs.io/en/2.x/get_started/installation.html).

- "Microsoft Visual C++ 14.0 or graeter is required" during installation on Windows.

  This error happens when building the 'pycocotools.\_mask' extension of pycocotools and the environment lacks corresponding C++ compilation dependencies. You need to download it at Microsoft officials [visual-cpp-build-tools](https://visualstudio.microsoft.com/zh-hans/visual-cpp-build-tools/),  select the "Use C ++ Desktop Development" option to install the minimum dependencies, and then reinstall pycocotools.

- Using Albumentations

  If you would like to use `albumentations`, we suggest using `pip install -r requirements/albu.txt` or
  `pip install -U albumentations --no-binary qudida,albumentations`.
  If you simply use `pip install albumentations>=0.3.2`, it will install `opencv-python-headless` simultaneously (even though you have already installed `opencv-python`).
  Please refer to the [official documentation](https://albumentations.ai/docs/getting_started/installation/#note-on-opencv-dependencies) for details.

- ModuleNotFoundError is raised when using some algorithms

  Some extra dependencies are required for Instaboost, Panoptic Segmentation, LVIS dataset, etc. Please note the error message and install corresponding packages, e.g.,

  ```shell
  # for instaboost
  pip install instaboostfast
  # for panoptic segmentation
  pip install git+https://github.com/cocodataset/panopticapi.git
  # for LVIS dataset
  pip install git+https://github.com/lvis-dataset/lvis-api.git
  ```

## Coding

- Do I need to reinstall mmdet after some code modifications

  If you follow the best practice and install mmdet with `pip install -e .`, any local modifications made to the code will take effect without reinstallation.

- How to develop with multiple MMDetection versions

  You can have multiple folders like mmdet-3.0, mmdet-3.1.
  When you run the train or test script, it will adopt the mmdet package in the current folder.

  To use the default MMDetection installed in the environment rather than the one you are working with, you can remove the following line in those scripts:

  ```shell
  PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
  ```

## PyTorch/CUDA Environment

- "RTX 30 series card fails when building MMCV or MMDet"

  1. Temporary work-around: do `MMCV_WITH_OPS=1 MMCV_CUDA_ARGS='-gencode=arch=compute_80,code=sm_80' pip install -e .`.
     The common issue is `nvcc fatal : Unsupported gpu architecture 'compute_86'`. This means that the compiler should optimize for sm_86, i.e., nvidia 30 series card, but such optimizations have not been supported by CUDA toolkit 11.0.
     This work-around modifies the compile flag by adding `MMCV_CUDA_ARGS='-gencode=arch=compute_80,code=sm_80'`, which tells `nvcc` to optimize for **sm_80**, i.e., Nvidia A100. Although A100 is different from the 30 series card, they use similar ampere architecture. This may hurt the performance but it works.
  2. PyTorch developers have updated that the default compiler flags should be fixed by [pytorch/pytorch#47585](https://github.com/pytorch/pytorch/pull/47585). So using PyTorch-nightly may also be able to solve the problem, though we have not tested it yet.

- "invalid device function" or "no kernel image is available for execution".

  1. Check if your cuda runtime version (under `/usr/local/`), `nvcc --version` and `conda list cudatoolkit` version match.
  2. Run `python mmdet/utils/collect_env.py` to check whether PyTorch, torchvision, and MMCV are built for the correct GPU architecture.
     You may need to set `TORCH_CUDA_ARCH_LIST` to reinstall MMCV.
     The GPU arch table could be found [here](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list),
     i.e. run `TORCH_CUDA_ARCH_LIST=7.0 pip install mmcv` to build MMCV for Volta GPUs.
     The compatibility issue could happen when using old GPUS, e.g., Tesla K80 (3.7) on colab.
  3. Check whether the running environment is the same as that when mmcv/mmdet has compiled.
     For example, you may compile mmcv using CUDA 10.0 but run it on CUDA 9.0 environments.

- "undefined symbol" or "cannot open xxx.so".

  1. If those symbols are CUDA/C++ symbols (e.g., libcudart.so or GLIBCXX), check whether the CUDA/GCC runtimes are the same as those used for compiling mmcv,
     i.e. run `python mmdet/utils/collect_env.py` to see if `"MMCV Compiler"`/`"MMCV CUDA Compiler"` is the same as `"GCC"`/`"CUDA_HOME"`.
  2. If those symbols are PyTorch symbols (e.g., symbols containing caffe, aten, and TH), check whether the PyTorch version is the same as that used for compiling mmcv.
  3. Run `python mmdet/utils/collect_env.py` to check whether PyTorch, torchvision, and MMCV are built by and running on the same environment.

- setuptools.sandbox.UnpickleableException: DistutilsSetupError("each element of 'ext_modules' option must be an Extension instance or 2-tuple")

  1. If you are using miniconda rather than anaconda, check whether Cython is installed as indicated in [#3379](https://github.com/open-mmlab/mmdetection/issues/3379).
     You need to manually install Cython first and then run command `pip install -r requirements.txt`.
  2. You may also need to check the compatibility between the `setuptools`, `Cython`, and `PyTorch` in your environment.

- "Segmentation fault".

  1. Check you GCC version and use GCC 5.4. This usually caused by the incompatibility between PyTorch and the environment (e.g., GCC \< 4.9 for PyTorch). We also recommend the users to avoid using GCC 5.5 because many feedbacks report that GCC 5.5 will cause "segmentation fault" and simply changing it to GCC 5.4 could solve the problem.

  2. Check whether PyTorch is correctly installed and could use CUDA op, e.g. type the following command in your terminal.

     ```shell
     python -c 'import torch; print(torch.cuda.is_available())'
     ```

     And see whether they could correctly output results.

  3. If Pytorch is correctly installed, check whether MMCV is correctly installed.

     ```shell
     python -c 'import mmcv; import mmcv.ops'
     ```

     If MMCV is correctly installed, then there will be no issue of the above two commands.

  4. If MMCV and Pytorch is correctly installed, you man use `ipdb`, `pdb` to set breakpoints or directly add 'print' in mmdetection code and see which part leads the segmentation fault.

## Training

- "Loss goes Nan"

  1. Check if the dataset annotations are valid: zero-size bounding boxes will cause the regression loss to be Nan due to the commonly used transformation for box regression. Some small size (width or height are smaller than 1) boxes will also cause this problem after data augmentation (e.g., instaboost). So check the data and try to filter out those zero-size boxes and skip some risky augmentations on the small-size boxes when you face the problem.
  2. Reduce the learning rate: the learning rate might be too large due to some reasons, e.g., change of batch size. You can rescale them to the value that could stably train the model.
  3. Extend the warmup iterations: some models are sensitive to the learning rate at the start of the training. You can extend the warmup iterations, e.g., change the `warmup_iters` from 500 to 1000 or 2000.
  4. Add gradient clipping: some models requires gradient clipping to stabilize the training process. The default of `grad_clip` is `None`, you can add gradient clippint to avoid gradients that are too large, i.e., set `optim_wrapper=dict(clip_grad=dict(max_norm=35, norm_type=2))` in your config file.

- "GPU out of memory"

  1. There are some scenarios when there are large amount of ground truth boxes, which may cause OOM during target assignment. You can set `gpu_assign_thr=N` in the config of assigner thus the assigner will calculate box overlaps through CPU when there are more than N GT boxes.

  2. Set `with_cp=True` in the backbone. This uses the sublinear strategy in PyTorch to reduce GPU memory cost in the backbone.

  3. Try mixed precision training using following the examples in `config/fp16`. The `loss_scale` might need further tuning for different models.

  4. Try to use `AvoidCUDAOOM` to avoid GPU out of memory. It will first retry after calling `torch.cuda.empty_cache()`. If it still fails, it will then retry by converting the type of inputs to FP16 format. If it still fails, it will try to copy inputs from GPUs to CPUs to continue computing. Try AvoidOOM in you code to make the code continue to run when GPU memory runs out:

     ```python
     from mmdet.utils import AvoidCUDAOOM

     output = AvoidCUDAOOM.retry_if_cuda_oom(some_function)(input1, input2)
     ```

     You can also try `AvoidCUDAOOM` as a decorator to make the code continue to run when GPU memory runs out:

     ```python
     from mmdet.utils import AvoidCUDAOOM

     @AvoidCUDAOOM.retry_if_cuda_oom
     def function(*args, **kwargs):
         ...
         return xxx
     ```

- "RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one"

  1. This error indicates that your module has parameters that were not used in producing loss. This phenomenon may be caused by running different branches in your code in DDP mode.
  2. You can set `find_unused_parameters = True` in the config to solve the above problems, but this will slow down the training speed.
  3. You can set `detect_anomalous_params = True` in the config or `model_wrapper_cfg = dict(type='MMDistributedDataParallel', detect_anomalous_params=True)` (More details please refer to [MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/model/wrappers/distributed.py#L91)) to get the name of those unused parameters. Note `detect_anomalous_params = True` will slow down the training speed, so it is recommended for debugging only.

- Save the best model

  It can be turned on by configuring `default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),`. In the case of the `auto` parameter, the first key in the returned evaluation result will be used as the basis for selecting the best model. You can also directly set the key in the evaluation result to manually set it, for example, `save_best='coco/bbox_mAP'`.

## Evaluation

- COCO Dataset, AP or AR = -1
  1. According to the definition of COCO dataset, the small and medium areas in an image are less than 1024 (32\*32), 9216 (96\*96), respectively.
  2. If the corresponding area has no object, the result of AP and AR will set to -1.

## Model

- `style` in ResNet

  The `style` parameter in ResNet allows either `pytorch` or `caffe` style. It indicates the difference in the Bottleneck module. Bottleneck is a stacking structure of `1x1-3x3-1x1` convolutional layers. In the case of `caffe` mode, the convolution layer with `stride=2` is the first `1x1` convolution, while in `pyorch` mode, it is the second `3x3` convolution has `stride=2`. A sample code is as below:

  ```python
  if self.style == 'pytorch':
        self.conv1_stride = 1
        self.conv2_stride = stride
  else:
        self.conv1_stride = stride
        self.conv2_stride = 1
  ```

- ResNeXt parameter description

  ResNeXt comes from the paper [`Aggregated Residual Transformations for Deep Neural Networks`](https://arxiv.org/abs/1611.05431). It introduces  group and uses “cardinality” to control the number of groups to achieve a balance between accuracy and complexity. It controls the basic width and grouping parameters of the internal Bottleneck module through two hyperparameters `baseWidth` and `cardinality`. An example configuration name in MMDetection is `mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco.py`, where `mask_rcnn` represents the algorithm using Mask R-CNN, `x101` represents the backbone network using ResNeXt-101, and `64x4d` represents that the bottleneck block has 64 group and each group has basic width of 4.

- `norm_eval` in backbone

  Since the detection model is usually large and the input image resolution is high, this will result in a small batch of the detection model, which will make the variance of the statistics calculated by BatchNorm during the training process very large and not as stable as the statistics obtained during the pre-training of the backbone network . Therefore, the `norm_eval=True` mode is generally used in training, and the BatchNorm statistics in the pre-trained backbone network are directly used. The few algorithms that use large batches are the `norm_eval=False` mode, such as NASFPN. For the backbone network without ImageNet pre-training and the batch is relatively small, you can consider using `SyncBN`.
