# 测试现有模型（待更新）

我们提供了测试脚本，能够测试一个现有模型在所有数据集（COCO，Pascal VOC，Cityscapes 等）上的性能。我们支持在如下环境下测试：

- 单 GPU 测试
- CPU 测试
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

# CPU 测试：禁用 GPU 并运行单 GPU 测试脚本
export CUDA_VISIBLE_DEVICES=-1
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

# CPU 测试：禁用 GPU 并运行单 GPU 测试脚本
export CUDA_VISIBLE_DEVICES=-1
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
