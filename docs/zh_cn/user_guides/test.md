# 测试现有模型

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
    [--show]

# CPU 测试：禁用 GPU 并运行单 GPU 测试脚本
export CUDA_VISIBLE_DEVICES=-1
python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--out ${RESULT_FILE}] \
    [--show]

# 单节点多 GPU 测试
bash tools/dist_test.sh \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${GPU_NUM} \
    [--out ${RESULT_FILE}]
```

`tools/dist_test.sh` 也支持多节点测试，不过需要依赖 PyTorch 的 [启动工具](https://pytorch.org/docs/stable/distributed.html#launch-utility) 。

可选参数：

- `RESULT_FILE`: 结果文件名称，需以 .pkl 形式存储。如果没有声明，则不将结果存储到文件。
- `--show`: 如果开启，检测结果将被绘制在图像上，以一个新窗口的形式展示。它只适用于单 GPU 的测试，是用于调试和可视化的。请确保使用此功能时，你的 GUI 可以在环境中打开。否则，你可能会遇到这么一个错误 `cannot connect to X server`。
- `--show-dir`: 如果指明，检测结果将会被绘制在图像上并保存到指定目录。它只适用于单 GPU 的测试，是用于调试和可视化的。即使你的环境中没有 GUI，这个选项也可使用。
- `--cfg-options`:  如果指明，这里的键值对将会被合并到配置文件中。

### 样例

假设你已经下载了 checkpoint 文件到 `checkpoints/` 文件下了。

1. 测试 RTMDet 并可视化其结果。按任意键继续下张图片的测试。配置文件和 checkpoint 文件 [在此](https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet) 。

   ```shell
   python tools/test.py \
       configs/rtmdet/rtmdet_l_8xb32-300e_coco.py \
       checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth \
       --show
   ```

2. 测试 RTMDet，并为了之后的可视化保存绘制的图像。配置文件和 checkpoint 文件 [在此](https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet) 。

   ```shell
   python tools/test.py \
       configs/rtmdet/rtmdet_l_8xb32-300e_coco.py \
       checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth \
       --show-dir rtmdet_l_8xb32-300e_coco_results
   ```

3. 在 Pascal VOC 数据集上测试 Faster R-CNN，不保存测试结果，测试 `mAP`。配置文件和 checkpoint 文件 [在此](../../../configs/pascal_voc) 。

   ```shell
   python tools/test.py \
       configs/pascal_voc/faster-rcnn_r50_fpn_1x_voc0712.py \
       checkpoints/faster_rcnn_r50_fpn_1x_voc0712_20200624-c9895d40.pth
   ```

4. 使用 8 块 GPU 测试 Mask R-CNN，测试 `bbox` 和 `mAP` 。配置文件和 checkpoint 文件 [在此](../../../configs/mask_rcnn) 。

   ```shell
   ./tools/dist_test.sh \
       configs/mask-rcnn_r50_fpn_1x_coco.py \
       checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth \
       8 \
       --out results.pkl
   ```

5. 使用 8 块 GPU 测试 Mask R-CNN，测试**每类**的 `bbox` 和 `mAP`。配置文件和 checkpoint 文件 [在此](../../../configs/mask_rcnn) 。

   ```shell
   ./tools/dist_test.sh \
       configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py \
       checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth \
       8
   ```

   该命令生成两个JSON文件 `./work_dirs/coco_instance/test.bbox.json` 和 `./work_dirs/coco_instance/test.segm.json`。

6. 在 COCO test-dev 数据集上，使用 8 块 GPU 测试 Mask R-CNN，并生成 JSON 文件提交到官方评测服务器，配置文件和 checkpoint 文件 [在此](../../../configs/mask_rcnnn) 。你可以在 [config](./././configs/_base_/datasets/coco_instance.py) 的注释中用 test_evaluator 和 test_dataloader 替换原来的 test_evaluator 和 test_dataloader，然后运行：

   ```shell
   ./tools/dist_test.sh \
       configs/cityscapes/mask-rcnn_r50_fpn_1x_cityscapes.py \
       checkpoints/mask_rcnn_r50_fpn_1x_cityscapes_20200227-afe51d5a.pth \
       8
   ```

   这行命令生成两个 JSON 文件 `mask_rcnn_test-dev_results.bbox.json` 和 `mask_rcnn_test-dev_results.segm.json`。

7. 在 Cityscapes 数据集上，使用 8 块 GPU 测试 Mask R-CNN，生成 txt 和 png 文件，并上传到官方评测服务器。配置文件和 checkpoint 文件 [在此](../../../configs/cityscapes) 。 你可以在 [config](./././configs/_base_/datasets/cityscapes_instance.py) 的注释中用 test_evaluator 和 test_dataloader 替换原来的 test_evaluator 和 test_dataloader，然后运行：

   ```shell
   ./tools/dist_test.sh \
       configs/cityscapes/mask-rcnn_r50_fpn_1x_cityscapes.py \
       checkpoints/mask_rcnn_r50_fpn_1x_cityscapes_20200227-afe51d5a.pth \
       8
   ```

   生成的 png 和 txt 文件在 `./work_dirs/cityscapes_metric` 文件夹下。

### 不使用 Ground Truth 标注进行测试

MMDetection 支持在不使用 ground-truth 标注的情况下对模型进行测试，这需要用到 `CocoDataset`。如果你的数据集格式不是 COCO 格式的，请将其转化成 COCO 格式。如果你的数据集格式是 VOC 或者 Cityscapes，你可以使用 [tools/dataset_converters](https://github.com/open-mmlab/mmdetection/tree/main/tools/dataset_converters) 内的脚本直接将其转化成 COCO 格式。如果是其他格式，可以使用 [images2coco 脚本](https://github.com/open-mmlab/mmdetection/tree/master/tools/dataset_converters/images2coco.py) 进行转换。

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
    [--show]

# CPU 测试：禁用 GPU 并运行单 GPU 测试脚本
export CUDA_VISIBLE_DEVICES=-1
python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--out ${RESULT_FILE}] \
    [--show]

# 单节点多 GPU 测试
bash tools/dist_test.sh \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${GPU_NUM} \
    [--show]
```

假设 [model zoo](https://mmdetection.readthedocs.io/en/latest/modelzoo_statistics.html) 中的 checkpoint 文件被下载到了 `checkpoints/` 文件夹下，
我们可以使用以下命令，用 8 块 GPU 在 COCO test-dev 数据集上测试 Mask R-CNN，并且生成 JSON 文件。

```sh
./tools/dist_test.sh \
    configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py \
    checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth \
    8
```

这行命令生成两个 JSON 文件 `./work_dirs/coco_instance/test.bbox.json` 和 `./work_dirs/coco_instance/test.segm.json`。

### 批量推理

MMDetection 在测试模式下，既支持单张图片的推理，也支持对图像进行批量推理。默认情况下，我们使用单张图片的测试，你可以通过修改测试数据配置文件中的 `samples_per_gpu` 来开启批量测试。
开启批量推理的配置文件修改方法为：

```shell
data = dict(train_dataloader=dict(...), val_dataloader=dict(...), test_dataloader=dict(batch_size=2, ...))
```

或者你可以通过将 `--cfg-options` 设置为 `--cfg-options test_dataloader.batch_size=` 来开启它。

## 测试时增强 (TTA)

测试时增强 (TTA) 是一种在测试阶段使用的数据增强策略。它对同一张图片应用不同的增强，例如翻转和缩放，用于模型推理，然后将每个增强后的图像的预测结果合并，以获得更准确的预测结果。为了让用户更容易使用 TTA，MMEngine 提供了 [BaseTTAModel](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.model.BaseTTAModel.html#mmengine.model.BaseTTAModel) 类，允许用户根据自己的需求通过简单地扩展 BaseTTAModel 类来实现不同的 TTA 策略。

在 MMDetection 中，我们提供了 [DetTTAModel](../../../mmdet/models/test_time_augs/det_tta.py) 类，它继承自 BaseTTAModel。

### 使用案例

使用 TTA 需要两个步骤。首先，你需要在配置文件中添加 `tta_model` 和 `tta_pipeline`：

```shell
tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(
                   type='nms',
                   iou_threshold=0.5),
                   max_per_img=100))

tta_pipeline = [
    dict(type='LoadImageFromFile',
        backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(type='Resize', scale=(1333, 800), keep_ratio=True)
        ], [ # It uses 2 flipping transformations (flipping and not flipping).
            dict(type='RandomFlip', prob=1.),
            dict(type='RandomFlip', prob=0.)
        ], [
            dict(
               type='PackDetInputs',
               meta_keys=('img_id', 'img_path', 'ori_shape',
                       'img_shape', 'scale_factor', 'flip',
                       'flip_direction'))
       ]])]
```

第二步，运行测试脚本时，设置 `--tta` 参数，如下所示：

```shell
# 单 GPU 测试
python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--tta]

# CPU 测试：禁用 GPU 并运行单 GPU 测试脚本
export CUDA_VISIBLE_DEVICES=-1
python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--out ${RESULT_FILE}] \
    [--tta]

# 多 GPU 测试
bash tools/dist_test.sh \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${GPU_NUM} \
    [--tta]
```

你也可以自己修改 TTA 配置，例如添加缩放增强：

```shell
tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(
                   type='nms',
                   iou_threshold=0.5),
                   max_per_img=100))

img_scales = [(1333, 800), (666, 400), (2000, 1200)]
tta_pipeline = [
    dict(type='LoadImageFromFile',
         backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(type='Resize', scale=s, keep_ratio=True) for s in img_scales
        ], [
            dict(type='RandomFlip', prob=1.),
            dict(type='RandomFlip', prob=0.)
        ], [
            dict(
               type='PackDetInputs',
               meta_keys=('img_id', 'img_path', 'ori_shape',
                       'img_shape', 'scale_factor', 'flip',
                       'flip_direction'))
       ]])]
```

以上数据增强管道将首先对图像执行 3 个多尺度转换，然后执行 2 个翻转转换（翻转和不翻转），最后使用 PackDetInputs 将图像打包到最终结果中。
这里有更多的 TTA 使用案例供您参考：

- [RetinaNet](../../../configs/retinanet/retinanet_tta.py)
- [CenterNet](../../../configs/centernet/centernet_tta.py)
- [YOLOX](../../../configs/rtmdet/rtmdet_tta.py)
- [RTMDet](../../../configs/yolox/yolox_tta.py)

更多高级用法和 TTA 的数据流，请参考 [MMEngine](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/test_time_augmentation.html#data-flow)。我们将在后续支持实例分割 TTA。
