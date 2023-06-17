# 在标准数据集上训练预定义的模型

MMDetection 也为训练检测模型提供了开盖即食的工具。本节将展示在标准数据集（比如 COCO）上如何训练一个预定义的模型。

### 数据集

训练需要准备好数据集，细节请参考 [数据集准备](#%E6%95%B0%E6%8D%AE%E9%9B%86%E5%87%86%E5%A4%87) 。

**注意**：
目前，`configs/cityscapes` 文件夹下的配置文件都是使用 COCO 预训练权值进行初始化的。如果网络连接不可用或者速度很慢，你可以提前下载现存的模型。否则可能在训练的开始会有错误发生。

### 学习率自动缩放

**注意**：在配置文件中的学习率是在 8 块 GPU，每块 GPU 有 2 张图像（批大小为 8\*2=16）的情况下设置的。其已经设置在 `config/_base_/schedules/schedule_1x.py` 中的 `auto_scale_lr.base_batch_size`。学习率会基于批次大小为 `16`时的值进行自动缩放。同时，为了不影响其他基于 mmdet 的 codebase，启用自动缩放标志 `auto_scale_lr.enable` 默认设置为 `False`。

如果要启用此功能，需在命令添加参数 `--auto-scale-lr`。并且在启动命令之前，请检查下即将使用的配置文件的名称，因为配置名称指示默认的批处理大小。
在默认情况下，批次大小是 `8 x 2 = 16`，例如：`faster_rcnn_r50_caffe_fpn_90k_coco.py` 或者 `pisa_faster_rcnn_x101_32x4d_fpn_1x_coco.py`；若不是默认批次，你可以在配置文件看到像 `_NxM_` 字样的，例如：`cornernet_hourglass104_mstest_32x3_210e_coco.py` 的批次大小是 `32 x 3 = 96`, 或者 `scnet_x101_64x4d_fpn_8x1_20e_coco.py` 的批次大小是 `8 x 1 = 8`。

**请记住：如果使用不是默认批次大小为 `16`的配置文件，请检查配置文件中的底部，会有 `auto_scale_lr.base_batch_size`。如果找不到，可以在其继承的 `_base_=[xxx]` 文件中找到。另外，如果想使用自动缩放学习率的功能，请不要修改这些值。**

学习率自动缩放基本用法如下：

```shell
python tools/train.py \
    ${CONFIG_FILE} \
    --auto-scale-lr \
    [optional arguments]
```

执行命令之后，会根据机器的GPU数量和训练的批次大小对学习率进行自动缩放，缩放方式详见 [线性扩展规则](https://arxiv.org/abs/1706.02677) ，比如：在 4 块 GPU 并且每张 GPU 上有 2 张图片的情况下 `lr=0.01`，那么在 16 块 GPU 并且每张 GPU 上有 4 张图片的情况下, LR 会自动缩放至 `lr=0.08`。

如果不启用该功能，则需要根据 [线性扩展规则](https://arxiv.org/abs/1706.02677) 来手动计算并修改配置文件里面 `optimizer.lr` 的值。

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
train_cfg = dict(val_interval=12)
```

这个工具接受以下参数：

- `--work-dir ${WORK_DIR}`: 覆盖工作目录.
- `--resume`：自动从work_dir中的最新检查点恢复.
- `--resume ${CHECKPOINT_FILE}`: 从某个 checkpoint 文件继续训练.
- `--cfg-options 'Key=value'`: 覆盖使用的配置文件中的其他设置.

**注意**：
`resume` 和 `load-from` 的区别：

`resume` 既加载了模型的权重和优化器的状态，也会继承指定 checkpoint 的迭代次数，不会重新开始训练。`load-from` 则是只加载模型的权重，它的训练是从头开始的，经常被用于微调模型。其中load-from需要写入配置文件中，而resume作为命令行参数传入。

### 使用 CPU 训练

使用 CPU 训练的流程和使用单 GPU 训练的流程一致，我们仅需要在训练流程开始前禁用 GPU。

```shell
export CUDA_VISIBLE_DEVICES=-1
```

之后运行单 GPU 训练脚本即可。

**注意**：

我们不推荐用户使用 CPU 进行训练，这太过缓慢。我们支持这个功能是为了方便用户在没有 GPU 的机器上进行调试。

### 在多 GPU 上训练

我们提供了 `tools/dist_train.sh` 来开启在多 GPU 上的训练。基本使用如下：

```shell
bash ./tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    [optional arguments]
```

可选参数和单 GPU 训练的可选参数一致。

#### 同时启动多个任务

如果你想在一台机器上启动多个任务的话，比如在一个有 8 块 GPU 的机器上启动 2 个需要 4 块GPU的任务，你需要给不同的训练任务指定不同的端口（默认为 29500）来避免冲突。

如果你使用 `dist_train.sh` 来启动训练任务，你可以使用命令来设置端口。

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```

### 使用多台机器训练

如果您想使用由 ethernet 连接起来的多台机器， 您可以使用以下命令:

在第一台机器上:

```shell
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR sh tools/dist_train.sh $CONFIG $GPUS
```

在第二台机器上:

```shell
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR sh tools/dist_train.sh $CONFIG $GPUS
```

但是，如果您不使用高速网路连接这几台机器的话，训练将会非常慢。

### 使用 Slurm 来管理任务

Slurm 是一个常见的计算集群调度系统。在 Slurm 管理的集群上，你可以使用 `slurm.sh` 来开启训练任务。它既支持单节点训练也支持多节点训练。

基本使用如下：

```shell
[GPUS=${GPUS}] ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR}
```

以下是在一个名称为 _dev_ 的 Slurm 分区上，使用 16 块 GPU 来训练 Mask R-CNN 的例子，并且将 `work-dir` 设置在了某些共享文件系统下。

```shell
GPUS=16 ./tools/slurm_train.sh dev mask_r50_1x configs/mask_rcnn_r50_fpn_1x_coco.py /nfs/xxxx/mask_rcnn_r50_fpn_1x
```

你可以查看 [源码](https://github.com/open-mmlab/mmdetection/blob/main/tools/slurm_train.sh) 来检查全部的参数和环境变量.

在使用 Slurm 时，端口需要以下方的某个方法之一来设置。

1. 通过 `--options` 来设置端口。我们非常建议用这种方法，因为它无需改变原始的配置文件。

   ```shell
   CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR} --cfg-options 'dist_params.port=29500'
   CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR} --cfg-options 'dist_params.port=29501'
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

# 在自定义数据集上进行训练

通过本文档，你将会知道如何使用自定义数据集对预先定义好的模型进行推理，测试以及训练。我们使用 [balloon dataset](https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon) 作为例子来描述整个过程。

基本步骤如下：

1. 准备自定义数据集
2. 准备配置文件
3. 在自定义数据集上进行训练，测试和推理。

## 准备自定义数据集

MMDetection 一共支持三种形式应用新数据集：

1. 将数据集重新组织为 COCO 格式。
2. 将数据集重新组织为一个中间格式。
3. 实现一个新的数据集。

我们通常建议使用前面两种方法，因为它们通常来说比第三种方法要简单。

在本文档中，我们展示一个例子来说明如何将数据转化为 COCO 格式。

**注意**：在 MMDetection 3.0 之后，数据集和指标已经解耦（除了 CityScapes）。因此，用户在验证阶段使用任意的评价指标来评价模型在任意数据集上的性能。比如，用 VOC 评价指标来评价模型在 COCO 数据集的性能，或者同时使用 VOC 评价指标和 COCO 评价指标来评价模型在 OpenImages 数据集上的性能。

### COCO标注格式

用于实例分割的 COCO 数据集格式如下所示，其中的键（key）都是必要的，参考[这里](https://cocodataset.org/#format-data)来获取更多细节。

```json
{
    "images": [image],
    "annotations": [annotation],
    "categories": [category]
}


image = {
    "id": int,
    "width": int,
    "height": int,
    "file_name": str,
}

annotation = {
    "id": int,
    "image_id": int,
    "category_id": int,
    "segmentation": RLE or [polygon],
    "area": float,
    "bbox": [x,y,width,height], # (x, y) 为 bbox 左上角的坐标
    "iscrowd": 0 or 1,
}

categories = [{
    "id": int,
    "name": str,
    "supercategory": str,
}]
```

现在假设我们使用 balloon dataset。

下载了数据集之后，我们需要实现一个函数将标注格式转化为 COCO 格式。然后我们就可以使用已经实现的 `CocoDataset` 类来加载数据并进行训练以及评测。

如果你浏览过新数据集，你会发现格式如下：

```json
{'base64_img_data': '',
 'file_attributes': {},
 'filename': '34020010494_e5cb88e1c4_k.jpg',
 'fileref': '',
 'regions': {'0': {'region_attributes': {},
   'shape_attributes': {'all_points_x': [1020,
     1000,
     994,
     1003,
     1023,
     1050,
     1089,
     1134,
     1190,
     1265,
     1321,
     1361,
     1403,
     1428,
     1442,
     1445,
     1441,
     1427,
     1400,
     1361,
     1316,
     1269,
     1228,
     1198,
     1207,
     1210,
     1190,
     1177,
     1172,
     1174,
     1170,
     1153,
     1127,
     1104,
     1061,
     1032,
     1020],
    'all_points_y': [963,
     899,
     841,
     787,
     738,
     700,
     663,
     638,
     621,
     619,
     643,
     672,
     720,
     765,
     800,
     860,
     896,
     942,
     990,
     1035,
     1079,
     1112,
     1129,
     1134,
     1144,
     1153,
     1166,
     1166,
     1150,
     1136,
     1129,
     1122,
     1112,
     1084,
     1037,
     989,
     963],
    'name': 'polygon'}}},
 'size': 1115004}
```

标注文件时是 JSON 格式的，其中所有键（key）组成了一张图片的所有标注。

其中将 balloon dataset 转化为 COCO 格式的代码如下所示。

```python
import os.path as osp

import mmcv

from mmengine.fileio import dump, load
from mmengine.utils import track_iter_progress


def convert_balloon_to_coco(ann_file, out_file, image_prefix):
    data_infos = load(ann_file)

    annotations = []
    images = []
    obj_count = 0
    for idx, v in enumerate(track_iter_progress(data_infos.values())):
        filename = v['filename']
        img_path = osp.join(image_prefix, filename)
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(
            dict(id=idx, file_name=filename, height=height, width=width))

        for _, obj in v['regions'].items():
            assert not obj['region_attributes']
            obj = obj['shape_attributes']
            px = obj['all_points_x']
            py = obj['all_points_y']
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            x_min, y_min, x_max, y_max = (min(px), min(py), max(px), max(py))

            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=0,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(x_max - x_min) * (y_max - y_min),
                segmentation=[poly],
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{
            'id': 0,
            'name': 'balloon'
        }])
    dump(coco_format_json, out_file)


if __name__ == '__main__':
    convert_balloon_to_coco(ann_file='data/balloon/train/via_region_data.json',
                            out_file='data/balloon/train/annotation_coco.json',
                            image_prefix='data/balloon/train')
    convert_balloon_to_coco(ann_file='data/balloon/val/via_region_data.json',
                            out_file='data/balloon/val/annotation_coco.json',
                            image_prefix='data/balloon/val')
```

使用如上的函数，用户可以成功将标注文件转化为 JSON 格式，之后可以使用 `CocoDataset` 对模型进行训练，并用 `CocoMetric` 评测。

## 准备配置文件

第二步需要准备一个配置文件来成功加载数据集。假设我们想要用 balloon dataset 来训练配备了 FPN 的 Mask R-CNN ，如下是我们的配置文件。假设配置文件命名为 `mask-rcnn_r50-caffe_fpn_ms-poly-1x_balloon.py`，相应保存路径为 `configs/balloon/`，配置文件内容如下所示。详细的配置文件方法可以参考[学习配置文件 — MMDetection 3.0.0 文档](https://mmdetection.readthedocs.io/zh_CN/latest/user_guides/config.html#base)。

```python
# 新配置继承了基本配置，并做了必要的修改
_base_ = '../mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-1x_coco.py'

# 我们还需要更改 head 中的 num_classes 以匹配数据集中的类别数
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1), mask_head=dict(num_classes=1)))

# 修改数据集相关配置
data_root = 'data/balloon/'
metainfo = {
    'classes': ('balloon', ),
    'palette': [
        (220, 20, 60),
    ]
}
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/annotation_coco.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val/annotation_coco.json',
        data_prefix=dict(img='val/')))
test_dataloader = val_dataloader

# 修改评价指标相关配置
val_evaluator = dict(ann_file=data_root + 'val/annotation_coco.json')
test_evaluator = val_evaluator

# 使用预训练的 Mask R-CNN 模型权重来做初始化，可以提高模型性能
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

```

## 训练一个新的模型

为了使用新的配置方法来对模型进行训练，你只需要运行如下命令。

```shell
python tools/train.py configs/balloon/mask-rcnn_r50-caffe_fpn_ms-poly-1x_balloon.py
```

参考 [在标准数据集上训练预定义的模型](https://mmdetection.readthedocs.io/zh_CN/latest/user_guides/train.html#id1) 来获取更多详细的使用方法。

## 测试以及推理

为了测试训练完毕的模型，你只需要运行如下命令。

```shell
python tools/test.py configs/balloon/mask-rcnn_r50-caffe_fpn_ms-poly-1x_balloon.py work_dirs/mask-rcnn_r50-caffe_fpn_ms-poly-1x_balloon/epoch_12.pth
```

参考 [测试现有模型](https://mmdetection.readthedocs.io/zh_CN/latest/user_guides/test.html) 来获取更多详细的使用方法。
