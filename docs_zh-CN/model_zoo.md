# 模型库

## 镜像地址

从MMDetection V2.0起，我们只通过阿里云维护模型库。V1.x版本的模型依然保存在亚马逊并且将会在未来逐渐弃用。

## 共同设置

- 所有模型都是在`coco_2017_train`上训练，在`coco_2017_val`上测试。
- 我们使用分布式训练。
- 所有pytorch-style的ImageNet预训练主干网络来自PyTorch的模型库，caffe-style的预训练主干网络来自detectron2最新开源的模型。
- 为了与其他代码库公平比较，文档中所写的GPU内存是8个GPU的`torch.cuda.max_memory_allocated()`的最大值，此值通常小于 nvidia-smi 显示的值。 
- 我们所写的模型推理时间包含网络前向时间和后处理时间，不包含数据加载时间。所有结果通过[benchmark.py](https://github.com/open-mmlab/mmdetection/blob/master/tools/analysis_tools/benchmark.py)脚本计算所得。该脚本会计算推理2000张图像的平均时间。

## ImageNet预训练模型

通过ImageNet分类任务预训练的主干网络进行初始化是很常见的操作。所有预训练模型的链接都可以在[open_mmlab](https://github.com/open-mmlab/mmcv/blob/master/mmcv/model_zoo/open_mmlab.json)中找到。根据`img_norm_cfg`和出处，我们可以将所有 ImageNet 预训练模型分为几种情况：

- TorchVision：torchvision模型权重，包含ResNet50, ResNet101。`img_norm_cfg` 为`dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)`。
- Pycls：[pycls](https://github.com/facebookresearch/pycls)模型权重，包含RegNetX。`img_norm_cfg`为`dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.12, 58.395], to_rgb=False)`。
- MSRA styles：[MSRA](https://github.com/KaimingHe/deep-residual-networks) 模型权重，包含ResNet50_Caffe，ResNet101_Caffe。`img_norm_cfg`为`dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)`。
- Caffe2 styles：现阶段只包含ResNext101_32x8d。`img_norm_cfg`为`dict(mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)`。
- Other styles: SSD的`img_norm_cfg`为`dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)`，YOLOv3的`img_norm_cfg`为`dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)`。

MMdetection常用到的主干网络细节如下表所示：

| 模型             | 来源         | 链接                                                         | 描述                                                  |
| ---------------- | ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ResNet50         | TorchVision | [torchvision's ResNet-50](https://download.pytorch.org/models/resnet50-19c8e357.pth) | 来自[torchvision's ResNet-50](https://download.pytorch.org/models/resnet50-19c8e357.pth)。 |
| ResNet101        | TorchVision | [torchvision's ResNet-101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth) | 来自[torchvision's ResNet-101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)。 |
| RegNetX          | Pycls       | [RegNetX_3.2gf](https://download.openmmlab.com/pretrain/third_party/regnetx_3.2gf-c2599b0f.pth)，[RegNetX_800mf](https://download.openmmlab.com/pretrain/third_party/regnetx_800mf-1f4be4c7.pth)等 | 来自[pycls](https://github.com/facebookresearch/pycls)。     |
| ResNet50_Caffe   | MSRA        | [MSRA's ResNet-50](https://download.openmmlab.com/pretrain/third_party/resnet50_caffe-788b5fa3.pth) | 由[Detectron2's R-50.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl)转化的副本。原始权重文件来自[MSRA's original ResNet-50](https://github.com/KaimingHe/deep-residual-networks)。|
| ResNet101_Caffe  | MSRA        | [MSRA's ResNet-101](https://download.openmmlab.com/pretrain/third_party/resnet101_caffe-3ad79236.pth) | 由[Detectron2's R-101.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl)转化的副本。原始权重文件来自[MSRA's original ResNet-101](https://github.com/KaimingHe/deep-residual-networks)。|
| ResNext101_32x8d | Caffe2      | [Caffe2 ResNext101_32x8d](https://download.openmmlab.com/pretrain/third_party/resnext101_32x8d-1516f1aa.pth) | 由[Detectron2's X-101-32x8d.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/FAIR/X-101-32x8d.pkl)转化的副本。原始ResNeXt-101-32x8d由FB使用Caffe2训练。|

## Baselines

### RPN

请参考[RPN](https://github.com/open-mmlab/mmdetection/blob/master/configs/rpn)。

### Faster R-CNN

请参考[Faster R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/faster_rcnn)。

### Mask R-CNN

请参考[Mask R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/mask_rcnn)。

### Fast R-CNN (使用提前计算的proposals)

请参考[Fast R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/fast_rcnn)。

### RetinaNet

请参考[RetinaNet](https://github.com/open-mmlab/mmdetection/blob/master/configs/retinanet)。

### Cascade R-CNN and Cascade Mask R-CNN

请参考[Cascade R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/cascade_rcnn)。

### Hybrid Task Cascade (HTC)

请参考[HTC](https://github.com/open-mmlab/mmdetection/blob/master/configs/htc)。

### SSD

请参考[SSD](https://github.com/open-mmlab/mmdetection/blob/master/configs/ssd)。

### Group Normalization (GN)

请参考[Group Normalization](https://github.com/open-mmlab/mmdetection/blob/master/configs/gn)。

### Weight Standardization

请参考[Weight Standardization](https://github.com/open-mmlab/mmdetection/blob/master/configs/gn+ws)。

### Deformable Convolution v2

请参考[Deformable Convolutional Networks](https://github.com/open-mmlab/mmdetection/blob/master/configs/dcn)。

### CARAFE: Content-Aware ReAssembly of FEatures

请参考[CARAFE](https://github.com/open-mmlab/mmdetection/blob/master/configs/carafe)。

### Instaboost

请参考[Instaboost](https://github.com/open-mmlab/mmdetection/blob/master/configs/instaboost)。

### Libra R-CNN

请参考[Libra R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/libra_rcnn)。

### Guided Anchoring

请参考[Guided Anchoring](https://github.com/open-mmlab/mmdetection/blob/master/configs/guided_anchoring)。

### FCOS

请参考[FCOS](https://github.com/open-mmlab/mmdetection/blob/master/configs/fcos)。

### FoveaBox

请参考[FoveaBox](https://github.com/open-mmlab/mmdetection/blob/master/configs/foveabox)。

### RepPoints

请参考[RepPoints](https://github.com/open-mmlab/mmdetection/blob/master/configs/reppoints)。

### FreeAnchor

请参考[FreeAnchor](https://github.com/open-mmlab/mmdetection/blob/master/configs/free_anchor)。

### Grid R-CNN (plus)

请参考[Grid R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/grid_rcnn)。

### GHM

请参考[GHM](https://github.com/open-mmlab/mmdetection/blob/master/configs/ghm)。

### GCNet

请参考[GCNet](https://github.com/open-mmlab/mmdetection/blob/master/configs/gcnet)。

### HRNet

请参考[HRNet](https://github.com/open-mmlab/mmdetection/blob/master/configs/hrnet)。

### Mask Scoring R-CNN

请参考[Mask Scoring R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/ms_rcnn)。

### Train from Scratch

请参考[Rethinking ImageNet Pre-training](https://github.com/open-mmlab/mmdetection/blob/master/configs/scratch)。

### NAS-FPN

请参考[NAS-FPN](https://github.com/open-mmlab/mmdetection/blob/master/configs/nas_fpn)。

### ATSS

请参考[ATSS](https://github.com/open-mmlab/mmdetection/blob/master/configs/atss)。

### FSAF

请参考[FSAF](https://github.com/open-mmlab/mmdetection/blob/master/configs/fsaf)。

### RegNetX

请参考[RegNet](https://github.com/open-mmlab/mmdetection/blob/master/configs/regnet)。

### Res2Net

请参考[Res2Net](https://github.com/open-mmlab/mmdetection/blob/master/configs/res2net)。

### GRoIE

请参考[GRoIE](https://github.com/open-mmlab/mmdetection/blob/master/configs/groie)。

### Dynamic R-CNN

请参考[Dynamic R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/dynamic_rcnn)。

### PointRend

请参考[PointRend](https://github.com/open-mmlab/mmdetection/blob/master/configs/point_rend)。

### DetectoRS

请参考[DetectoRS](https://github.com/open-mmlab/mmdetection/blob/master/configs/detectors)。

### Generalized Focal Loss

请参考[Generalized Focal Loss](https://github.com/open-mmlab/mmdetection/blob/master/configs/gfl)。

### CornerNet

请参考[CornerNet](https://github.com/open-mmlab/mmdetection/blob/master/configs/cornernet)。

### YOLOv3

请参考[YOLOv3](https://github.com/open-mmlab/mmdetection/blob/master/configs/yolo)。

### PAA

请参考[PAA](https://github.com/open-mmlab/mmdetection/blob/master/configs/paa)。

### SABL

请参考[SABL](https://github.com/open-mmlab/mmdetection/blob/master/configs/sabl)。

### CentripetalNet

请参考[CentripetalNet](https://github.com/open-mmlab/mmdetection/blob/master/configs/centripetalnet)。
### ResNeSt

请参考[ResNeSt](https://github.com/open-mmlab/mmdetection/blob/master/configs/resnest)。

### DETR

请参考[DETR](https://github.com/open-mmlab/mmdetection/blob/master/configs/detr)。

### Deformable DETR

请参考[Deformable DETR](https://github.com/open-mmlab/mmdetection/blob/master/configs/deformable_detr)。

### AutoAssign

请参考[AutoAssign](https://github.com/open-mmlab/mmdetection/blob/master/configs/autoassign)。

### YOLOF

请参考[YOLOF](https://github.com/open-mmlab/mmdetection/blob/master/configs/yolof)。


### Seesaw Loss

请参考[Seesaw Loss](https://github.com/open-mmlab/mmdetection/blob/master/configs/seesaw_loss)。

### CenterNet

请参考[CenterNet](https://github.com/open-mmlab/mmdetection/blob/master/configs/centernet)。

### Other datasets

我们还在[PASCAL VOC](https://github.com/open-mmlab/mmdetection/blob/master/configs/pascal_voc)，[Cityscapes](https://github.com/open-mmlab/mmdetection/blob/master/configs/cityscapes)和[WIDER FACE](https://github.com/open-mmlab/mmdetection/blob/master/configs/wider_face)上对一些方法进行了基准测试。

### Pre-trained Models

我们还通过多尺度训练、更长的训练策略来训练用ResNet-50和[RegNetX-3.2G](https://github.com/open-mmlab/mmdetection/blob/master/configs/regnet)作为主干网络的[Faster R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/faster_rcnn)和[Mask R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/mask_rcnn)。这些模型可以作为下游任务的预训练模型。