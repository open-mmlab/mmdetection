# 1: 使用已有模型在标准数据集上进行推理
MMDetection 在 [Model Zoo](https://mmdetection.readthedocs.io/en/latest/model_zoo.html) 中提供了数以百计的检测模型，并支持多种标准数据集，包括 Pascal VOC，COCO，Cityscapes，LVIS 等。这份文档将会讲述如何使用这些模型和标准数据集来运行一些常见的任务，包括：

- 使用现有模型在给定图片上进行推理
- 在标准数据集上测试现有模型
- 在标准数据集上训练预定义的模型

## 使用现有模型进行推理

推理是指使用训练好的模型来检测图像上的目标。在 MMDetection 中，一个模型被定义为一个配置文件和对应的存储在 checkpoint 文件内的模型参数的集合。

首先，我们建议从 [Faster RCNN](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn) 开始，其 [配置](https://github.com/open-mmlab/mmdetection/blob/master/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py) 文件和 [checkpoint](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth) 文件在此。
我们建议将 checkpoint 文件下载到 `checkpoints` 文件夹内。

### 推理的高层编程接口

MMDetection 为在图片上推理提供了 Python 的高层编程接口。下面是建立模型和在图像或视频上进行推理的例子。

```python
from mmdet.apis import init_detector, inference_detector
import mmcv

# 指定模型的配置文件和 checkpoint 文件路径
config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# 根据配置文件和 checkpoint 文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 测试单张图片并展示结果
img = 'test.jpg'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
result = inference_detector(model, img)
# 在一个新的窗口中将结果可视化
model.show_result(img, result)
# 或者将可视化结果保存为图片
model.show_result(img, result, out_file='result.jpg')

# 测试视频并展示结果
video = mmcv.VideoReader('video.mp4')
for frame in video:
    result = inference_detector(model, frame)
    model.show_result(frame, result, wait_time=1)
```

jupyter notebook 上的演示样例在 [demo/inference_demo.ipynb](https://github.com/open-mmlab/mmdetection/blob/master/demo/inference_demo.ipynb) 。

### 异步接口-支持 Python 3.7+
对于 Python 3.7+，MMDetection 也有异步接口。利用 CUDA 流，绑定 GPU 的推理代码不会阻塞 CPU，从而使得 CPU/GPU 在单线程应用中能达到更高的利用率。在推理流程中，不同数据样本的推理和不同模型的推理都能并发地运行。

您可以参考 `tests/async_benchmark.py` 来对比同步接口和异步接口的运行速度。

```python
import asyncio
import torch
from mmdet.apis import init_detector, async_inference_detector
from mmdet.utils.contextmanagers import concurrent

async def main():
    config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    device = 'cuda:0'
    model = init_detector(config_file, checkpoint=checkpoint_file, device=device)

    # 此队列用于并行推理多张图像
    streamqueue = asyncio.Queue()
    # 队列大小定义了并行的数量
    streamqueue_size = 3

    for _ in range(streamqueue_size):
        streamqueue.put_nowait(torch.cuda.Stream(device=device))

    # 测试单张图片并展示结果
    img = 'test.jpg'  # or 或者 img = mmcv.imread(img)，这样图片仅会被读一次

    async with concurrent(streamqueue):
        result = await async_inference_detector(model, img)

    # 在一个新的窗口中将结果可视化
    model.show_result(img, result)
    # 或者将可视化结果保存为图片
    model.show_result(img, result, out_file='result.jpg')


asyncio.run(main())

```

### 演示样例
我们还提供了三个演示脚本，它们是使用高层编程接口实现的。 [源码在此](https://github.com/open-mmlab/mmdetection/tree/master/demo) 。

#### 图片样例
这是在单张图片上进行推理的脚本，可以开启 `--async-test` 来进行异步推理。

   ```shell
   python demo/image_demo.py \
       ${IMAGE_FILE} \
       ${CONFIG_FILE} \
       ${CHECKPOINT_FILE} \
       [--device ${GPU_ID}] \
       [--score-thr ${SCORE_THR}] \
       [--async-test]
   ```

运行样例：

   ```shell
   python demo/image_demo.py demo/demo.jpg \
       configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
       checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
       --device cpu
   ```

#### 摄像头样例
这是使用摄像头实时图片的推理脚本。

   ```shell
   python demo/webcam_demo.py \
       ${CONFIG_FILE} \
       ${CHECKPOINT_FILE} \
       [--device ${GPU_ID}] \
       [--camera-id ${CAMERA-ID}] \
       [--score-thr ${SCORE_THR}]
   ```

运行样例：

   ```shell
   python demo/webcam_demo.py \
       configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
       checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
   ```

#### 视频样例
这是在视频样例上进行推理的脚本。

   ```shell
   python demo/video_demo.py \
       ${VIDEO_FILE} \
       ${CONFIG_FILE} \
       ${CHECKPOINT_FILE} \
       [--device ${GPU_ID}] \
       [--score-thr ${SCORE_THR}] \
       [--out ${OUT_FILE}] \
       [--show] \
       [--wait-time ${WAIT_TIME}]
   ```

运行样例：

   ```shell
   python demo/video_demo.py demo/demo.mp4 \
       configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
       checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
       --out result.mp4
   ```

## 在标准数据集上测试现有模型
为了测试一个模型的精度，我们通常会在标准数据集上对其进行测试。MMDetection 支持多个公共数据集，包括 [COCO](https://cocodataset.org/) ，
[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC) ，[Cityscapes](https://www.cityscapes-dataset.com/) 等等。
这一部分将会介绍如何在支持的数据集上测试现有模型。

### 数据集准备
一些公共数据集，比如 Pascal VOC 及其镜像数据集，或者 COCO 等数据集都可以从官方网站或者镜像网站获取。
注意：在检测任务中，Pascal VOC 2012 是 Pascal VOC 2007 的无交集扩展，我们通常将两者一起使用。
我们建议将数据集下载，然后解压到项目外部的某个文件夹内，然后通过符号链接的方式，将数据集根目录链接到 `$MMDETECTION/data` 文件夹下，格式如下所示。
如果你的文件夹结构和下方不同的话，你需要在配置文件中改变对应的路径。

```plain
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
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   ├── VOC2012
```

有些模型需要额外的 [COCO-stuff](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip) 数据集，比如 HTC，DetectoRS 和 SCNet，你可以下载并解压它们到 `coco` 文件夹下。文件夹会是如下结构：

```plain
mmdetection
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   │   ├── stuffthingmaps
```

PanopticFPN 等全景分割模型需要额外的 [COCO Panoptic](http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip) 数据集，你可以下载并解压它们到 `coco/annotations` 文件夹下。文件夹会是如下结构：

```text
mmdetection
├── data
│   ├── coco
│   │   ├── annotations
│   │   │   ├── panoptic_train2017.json
│   │   │   ├── panoptic_train2017
│   │   │   ├── panoptic_val2017.json
│   │   │   ├── panoptic_val2017
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```

Cityscape 数据集的标注格式需要转换，以与 COCO 数据集标注格式保持一致，使用 `tools/dataset_converters/cityscapes.py` 来完成转换：

```shell
pip install cityscapesscripts

python tools/dataset_converters/cityscapes.py \
    ./data/cityscapes \
    --nproc 8 \
    --out-dir ./data/cityscapes/annotations
```

### 测试现有模型
我们提供了测试脚本，能够测试一个现有模型在所有数据集（COCO，Pascal VOC，Cityscapes 等）上的性能。我们支持在如下环境下测试：

- 单 GPU 测试
- 单节点多 GPU 测试
- 多节点测试

根据以上测试环境，选择合适的脚本来执行测试过程。

```shell
# 单 GPU 测试
python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--out ${RESULT_FILE}] \
    [--eval ${EVAL_METRICS}] \
    [--show]

# 单节点多 GPU 测试
bash tools/dist_test.sh \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${GPU_NUM} \
    [--out ${RESULT_FILE}] \
    [--eval ${EVAL_METRICS}]
```

`tools/dist_test.sh` 也支持多节点测试，不过需要依赖 PyTorch 的 [启动工具](https://pytorch.org/docs/stable/distributed.html#launch-utility) 。

可选参数：

- `RESULT_FILE`: 结果文件名称，需以 .pkl 形式存储。如果没有声明，则不将结果存储到文件。
- `EVAL_METRICS`: 需要测试的度量指标。可选值是取决于数据集的，比如 `proposal_fast`，`proposal`，`bbox`，`segm` 是 COCO 数据集的可选值，`mAP`，`recall` 是 Pascal VOC 数据集的可选值。Cityscapes 数据集可以测试 `cityscapes` 和所有 COCO 数据集支持的度量指标。
- `--show`: 如果开启，检测结果将被绘制在图像上，以一个新窗口的形式展示。它只适用于单 GPU 的测试，是用于调试和可视化的。请确保使用此功能时，你的 GUI 可以在环境中打开。否则，你可能会遇到这么一个错误 `cannot connect to X server`。
- `--show-dir`: 如果指明，检测结果将会被绘制在图像上并保存到指定目录。它只适用于单 GPU 的测试，是用于调试和可视化的。即使你的环境中没有 GUI，这个选项也可使用。
- `--show-score-thr`: 如果指明，得分低于此阈值的检测结果将会被移除。
- `--cfg-options`:  如果指明，这里的键值对将会被合并到配置文件中。
- `--eval-options`: 如果指明，这里的键值对将会作为字典参数被传入 `dataset.evaluation()` 函数中，仅在测试阶段使用。

### 样例

假设你已经下载了 checkpoint 文件到 `checkpoints/` 文件下了。

1. 测试 Faster R-CNN 并可视化其结果。按任意键继续下张图片的测试。配置文件和 checkpoint 文件 [在此](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn) 。

   ```shell
   python tools/test.py \
       configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
       checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
       --show
   ```

2. 测试 Faster R-CNN，并为了之后的可视化保存绘制的图像。配置文件和 checkpoint 文件 [在此](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn) 。

   ```shell
   python tools/test.py \
       configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
       checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
       --show-dir faster_rcnn_r50_fpn_1x_results
   ```

3. 在 Pascal VOC 数据集上测试 Faster R-CNN，不保存测试结果，测试 `mAP`。配置文件和 checkpoint 文件 [在此](https://github.com/open-mmlab/mmdetection/tree/master/configs/pascal_voc) 。

   ```shell
   python tools/test.py \
       configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc.py \
       checkpoints/faster_rcnn_r50_fpn_1x_voc0712_20200624-c9895d40.pth \
       --eval mAP
   ```

4. 使用 8 块 GPU 测试 Mask R-CNN，测试 `bbox` 和 `mAP` 。配置文件和 checkpoint 文件 [在此](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn) 。

   ```shell
   ./tools/dist_test.sh \
       configs/mask_rcnn_r50_fpn_1x_coco.py \
       checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth \
       8 \
       --out results.pkl \
       --eval bbox segm
   ```

5. 使用 8 块 GPU 测试 Mask R-CNN，测试**每类**的 `bbox` 和 `mAP`。配置文件和 checkpoint 文件 [在此](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn) 。

   ```shell
   ./tools/dist_test.sh \
       configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py \
       checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth \
       8 \
       --out results.pkl \
       --eval bbox segm \
       --options "classwise=True"
   ```

6. 在 COCO test-dev 数据集上，使用 8 块 GPU 测试 Mask R-CNN，并生成 JSON 文件提交到官方评测服务器。配置文件和 checkpoint 文件 [在此](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn) 。

   ```shell
   ./tools/dist_test.sh \
       configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py \
       checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth \
       8 \
       --format-only \
       --options "jsonfile_prefix=./mask_rcnn_test-dev_results"
   ```

这行命令生成两个 JSON 文件 `mask_rcnn_test-dev_results.bbox.json` 和 `mask_rcnn_test-dev_results.segm.json`。

7. 在 Cityscapes 数据集上，使用 8 块 GPU 测试 Mask R-CNN，生成 txt 和 png 文件，并上传到官方评测服务器。配置文件和 checkpoint 文件 [在此](https://github.com/open-mmlab/mmdetection/tree/master/configs/cityscapes) 。

   ```shell
   ./tools/dist_test.sh \
       configs/cityscapes/mask_rcnn_r50_fpn_1x_cityscapes.py \
       checkpoints/mask_rcnn_r50_fpn_1x_cityscapes_20200227-afe51d5a.pth \
       8 \
       --format-only \
       --options "txtfile_prefix=./mask_rcnn_cityscapes_test_results"
   ```

生成的 png 和 txt 文件在 `./mask_rcnn_cityscapes_test_results` 文件夹下。

### 不使用 Ground Truth 标注进行测试
MMDetection 支持在不使用 ground-truth 标注的情况下对模型进行测试，这需要用到 `CocoDataset`。如果你的数据集格式不是 COCO 格式的，请将其转化成 COCO 格式。如果你的数据集格式是 VOC 或者 Cityscapes，你可以使用 [tools/dataset_converters](https://github.com/open-mmlab/mmdetection/tree/master/tools/dataset_converters) 内的脚本直接将其转化成 COCO 格式。如果是其他格式，可以使用 [images2coco 脚本](https://github.com/open-mmlab/mmdetection/tree/master/tools/dataset_converters/images2coco.py) 进行转换。

```shell
python tools/dataset_converters/images2coco.py \
    ${IMG_PATH} \
    ${CLASSES} \
    ${OUT} \
    [--exclude-extensions]
```

参数：

- `IMG_PATH`: 图片根路径。
- `CLASSES`: 类列表文本文件名。文本中每一行存储一个类别。
- `OUT`: 输出 json 文件名。 默认保存目录和 `IMG_PATH` 在同一级。
- `exclude-extensions`: 待排除的文件后缀名。


在转换完成后，使用如下命令进行测试

```shell
# 单 GPU 测试
python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    --format-only \
    --options ${JSONFILE_PREFIX} \
    [--show]

# 单节点多 GPU 测试
bash tools/dist_test.sh \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${GPU_NUM} \
    --format-only \
    --options ${JSONFILE_PREFIX} \
    [--show]
```

假设 [model zoo](https://mmdetection.readthedocs.io/en/latest/modelzoo_statistics.html) 中的 checkpoint 文件被下载到了 `checkpoints/` 文件夹下，
我们可以使用以下命令，用 8 块 GPU 在 COCO test-dev 数据集上测试 Mask R-CNN，并且生成 JSON 文件。

```sh
./tools/dist_test.sh \
    configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py \
    checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth \
    8 \
    -format-only \
    --options "jsonfile_prefix=./mask_rcnn_test-dev_results"
```

这行命令生成两个 JSON 文件 `mask_rcnn_test-dev_results.bbox.json` 和 `mask_rcnn_test-dev_results.segm.json`。

### 批量推理

MMDetection 在测试模式下，既支持单张图片的推理，也支持对图像进行批量推理。默认情况下，我们使用单张图片的测试，你可以通过修改测试数据配置文件中的 `samples_per_gpu` 来开启批量测试。
开启批量推理的配置文件修改方法为：

```shell
data = dict(train=dict(...), val=dict(...), test=dict(samples_per_gpu=2, ...))
```

或者你可以通过将 `--cfg-options` 设置为 `--cfg-options data.test.samples_per_gpu=2` 来开启它。

### 弃用 ImageToTensor
在测试模式下，弃用 `ImageToTensor` 流程，取而代之的是 `DefaultFormatBundle`。建议在你的测试数据流程的配置文件中手动替换它，如：

```python
# （已弃用）使用 ImageToTensor
pipelines = [
   dict(type='LoadImageFromFile'),
   dict(
       type='MultiScaleFlipAug',
       img_scale=(1333, 800),
       flip=False,
       transforms=[
           dict(type='Resize', keep_ratio=True),
           dict(type='RandomFlip'),
           dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1]),
           dict(type='Pad', size_divisor=32),
           dict(type='ImageToTensor', keys=['img']),
           dict(type='Collect', keys=['img']),
       ])
   ]

# （建议使用）手动将 ImageToTensor 替换为 DefaultFormatBundle
pipelines = [
   dict(type='LoadImageFromFile'),
   dict(
       type='MultiScaleFlipAug',
       img_scale=(1333, 800),
       flip=False,
       transforms=[
           dict(type='Resize', keep_ratio=True),
           dict(type='RandomFlip'),
           dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1]),
           dict(type='Pad', size_divisor=32),
           dict(type='DefaultFormatBundle'),
           dict(type='Collect', keys=['img']),
       ])
   ]
```

## 在标准数据集上训练预定义的模型

MMDetection 也为训练检测模型提供了开盖即食的工具。本节将展示在标准数据集（比如 COCO）上如何训练一个预定义的模型。

重要信息：在配置文件中的学习率是在 8 块 GPU，每块 GPU 有 2 张图像（批大小为 8*2=16）的情况下设置的。
根据 [线性扩展规则](https://arxiv.org/abs/1706.02677) ，如果你使用不同数目的 GPU 或者每块 GPU 上有不同数量的图片，你需要设置学习率以正比于批大小，比如，
在 4 块 GPU 并且每张 GPU 上有 2 张图片的情况下，设置 `lr=0.01`； 在 16 块 GPU 并且每张 GPU 上有 4 张图片的情况下, 设置 `lr=0.08`。

### 数据集
训练需要准备好数据集，细节请参考 [数据集准备](#数据集准备) 。

**注意**：
目前，`configs/cityscapes` 文件夹下的配置文件都是使用 COCO 预训练权值进行初始化的。如果网络连接不可用或者速度很慢，你可以提前下载现存的模型。否则可能在训练的开始会有错误发生。

### 使用单 GPU 训练
我们提供了 `tools/train.py` 来开启在单张 GPU 上的训练任务。基本使用如下：

```shell
python tools/train.py \
    ${CONFIG_FILE} \
    [optional arguments]
```

在训练期间，日志文件和 checkpoint 文件将会被保存在工作目录下，它需要通过配置文件中的 `work_dir` 或者 CLI 参数中的 `--work-dir` 来指定。

默认情况下，模型将在每轮训练之后在 validation 集上进行测试，测试的频率可以通过设置配置文件来指定：

```python
# 每 12 轮迭代进行一次测试评估
evaluation = dict(interval=12)
```
这个工具接受以下参数：

- `--no-validate` (**不建议**): 在训练期间关闭测试.
- `--work-dir ${WORK_DIR}`: 覆盖工作目录.
- `--resume-from ${CHECKPOINT_FILE}`: 从某个 checkpoint 文件继续训练.
- `--options 'Key=value'`: 覆盖使用的配置文件中的其他设置.

**注意**：
`resume-from` 和 `load-from` 的区别：

`resume-from` 既加载了模型的权重和优化器的状态，也会继承指定 checkpoint 的迭代次数，不会重新开始训练。`load-from` 则是只加载模型的权重，它的训练是从头开始的，经常被用于微调模型。

### 在多 GPU 上训练
我们提供了 `tools/dist_train.sh` 来开启在多 GPU 上的训练。基本使用如下：

```shell
bash ./tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    [optional arguments]
```

可选参数和上一节所说的一致。

#### 同时启动多个任务
如果你想在一台机器上启动多个任务的话，比如在一个有 8 块 GPU 的机器上启动 2 个需要 4 块GPU的任务，你需要给不同的训练任务指定不同的端口（默认为 29500）来避免冲突。

如果你使用 `dist_train.sh` 来启动训练任务，你可以使用命令来设置端口。

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```

#### 在多个节点上训练
MMDetection 是依赖 `torch.distributed` 包进行分布式训练的。因此，我们可以通过 PyTorch 的 [启动工具](https://pytorch.org/docs/stable/distributed.html#launch-utility) 来进行基本地使用。

#### 使用 Slurm 来管理任务

Slurm 是一个常见的计算集群调度系统。在 Slurm 管理的集群上，你可以使用 `slurm.sh` 来开启训练任务。它既支持单节点训练也支持多节点训练。

基本使用如下：

```shell
[GPUS=${GPUS}] ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR}
```

以下是在一个名称为 _dev_ 的 Slurm 分区上，使用 16 块 GPU 来训练 Mask R-CNN 的例子，并且将 `work-dir` 设置在了某些共享文件系统下。

```shell
GPUS=16 ./tools/slurm_train.sh dev mask_r50_1x configs/mask_rcnn_r50_fpn_1x_coco.py /nfs/xxxx/mask_rcnn_r50_fpn_1x
```

你可以查看 [源码](https://github.com/open-mmlab/mmdetection/blob/master/tools/slurm_train.sh) 来检查全部的参数和环境变量.

在使用 Slurm 时，端口需要以下方的某个方法之一来设置。

1. 通过 `--options` 来设置端口。我们非常建议用这种方法，因为它无需改变原始的配置文件。

   ```shell
   CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR} --options 'dist_params.port=29500'
   CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR} --options 'dist_params.port=29501'
   ```

2. 修改配置文件来设置不同的交流端口。

   在 `config1.py` 中，设置：

   ```python
   dist_params = dict(backend='nccl', port=29500)
   ```

   在 `config2.py` 中，设置：

   ```python
   dist_params = dict(backend='nccl', port=29501)
   ```

   然后你可以使用 `config1.py` 和 `config2.py` 来启动两个任务了。

   ```shell
   CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR}
   CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR}
   ```
