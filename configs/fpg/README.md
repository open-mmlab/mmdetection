# FPG

> [Feature Pyramid Grids](https://arxiv.org/abs/2004.03580)

<!-- [ALGORITHM] -->

## Abstract

Feature pyramid networks have been widely adopted in the object detection literature to improve feature representations for better handling of variations in scale. In this paper, we present Feature Pyramid Grids (FPG), a deep multi-pathway feature pyramid, that represents the feature scale-space as a regular grid of parallel bottom-up pathways which are fused by multi-directional lateral connections. FPG can improve single-pathway feature pyramid networks by significantly increasing its performance at similar computation cost, highlighting importance of deep pyramid representations. In addition to its general and uniform structure, over complicated structures that have been found with neural architecture search, it also compares favorably against such approaches without relying on search. We hope that FPG with its uniform and effective nature can serve as a strong component for future work in object recognition.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143885611-85902399-2885-4a85-9126-9b9b7464ad08.png"/>
</div>

## Results and Models

We benchmark the new training schedule (crop training, large batch, unfrozen BN, 50 epochs) introduced in NAS-FPN.
All backbones are Resnet-50 in pytorch style.

|    Method    |    Neck    | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP |                           Config                           |                                                                                                                                                                             Download                                                                                                                                                                             |
| :----------: | :--------: | :-----: | :------: | :------------: | :----: | :-----: | :--------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Faster R-CNN |    FPG     |   50e   |   20.0   |       -        |  42.3  |    -    |    [config](./faster-rcnn_r50_fpg_crop640-50e_coco.py)     |        [model](https://download.openmmlab.com/mmdetection/v2.0/fpg/faster_rcnn_r50_fpg_crop640_50e_coco/faster_rcnn_r50_fpg_crop640_50e_coco_20220311_011856-74109f42.pth) \|               [log](https://download.openmmlab.com/mmdetection/v2.0/fpg/faster_rcnn_r50_fpg_crop640_50e_coco/faster_rcnn_r50_fpg_crop640_50e_coco_20220311_011856.log.json)        |
| Faster R-CNN | FPG-chn128 |   50e   |   11.9   |       -        |  41.2  |    -    | [config](./faster-rcnn_r50_fpg-chn128_crop640-50e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/fpg/faster_rcnn_r50_fpg-chn128_crop640_50e_coco/faster_rcnn_r50_fpg-chn128_crop640_50e_coco_20220311_011857-9376aa9d.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/fpg/faster_rcnn_r50_fpg-chn128_crop640_50e_coco/faster_rcnn_r50_fpg-chn128_crop640_50e_coco_20220311_011857.log.json) |
| Faster R-CNN |    FPN     |   50e   |   20.0   |       -        |  38.9  |    -    |    [config](./faster-rcnn_r50_fpn_crop640-50e_coco.py)     |        [model](https://download.openmmlab.com/mmdetection/v2.0/fpg/faster_rcnn_r50_fpn_crop640_50e_coco/faster_rcnn_r50_fpn_crop640_50e_coco_20220311_011857-be7c9f42.pth) \|               [log](https://download.openmmlab.com/mmdetection/v2.0/fpg/faster_rcnn_r50_fpn_crop640_50e_coco/faster_rcnn_r50_fpn_crop640_50e_coco_20220311_011857.log.json)        |
|  Mask R-CNN  |    FPG     |   50e   |   23.2   |       -        |  43.0  |  38.1   |     [config](./mask-rcnn_r50_fpg_crop640-50e_coco.py)      |          [model](https://download.openmmlab.com/mmdetection/v2.0/fpg/mask_rcnn_r50_fpg_crop640_50e_coco/mask_rcnn_r50_fpg_crop640_50e_coco_20220311_011857-233b8334.pth) \|                   [log](https://download.openmmlab.com/mmdetection/v2.0/fpg/mask_rcnn_r50_fpg_crop640_50e_coco/mask_rcnn_r50_fpg_crop640_50e_coco_20220311_011857.log.json)          |
|  Mask R-CNN  | FPG-chn128 |   50e   |   15.3   |       -        |  41.7  |  37.1   |  [config](./mask-rcnn_r50_fpg-chn128_crop640-50e_coco.py)  |   [model](https://download.openmmlab.com/mmdetection/v2.0/fpg/mask_rcnn_r50_fpg-chn128_crop640_50e_coco/mask_rcnn_r50_fpg-chn128_crop640_50e_coco_20220311_011859-043c9b4e.pth) \|     [log](https://download.openmmlab.com/mmdetection/v2.0/fpg/mask_rcnn_r50_fpg-chn128_crop640_50e_coco/mask_rcnn_r50_fpg-chn128_crop640_50e_coco_20220311_011859.log.json)   |
|  Mask R-CNN  |    FPN     |   50e   |   23.2   |       -        |  49.6  |  35.6   |     [config](./mask-rcnn_r50_fpn_crop640-50e_coco.py)      |          [model](https://download.openmmlab.com/mmdetection/v2.0/fpg/mask_rcnn_r50_fpn_crop640_50e_coco/mask_rcnn_r50_fpn_crop640_50e_coco_20220311_011855-a756664a.pth) \|                   [log](https://download.openmmlab.com/mmdetection/v2.0/fpg/mask_rcnn_r50_fpn_crop640_50e_coco/mask_rcnn_r50_fpn_crop640_50e_coco_20220311_011855.log.json)          |
|  RetinaNet   |    FPG     |   50e   |   20.8   |       -        |  40.5  |    -    |     [config](./retinanet_r50_fpg_crop640_50e_coco.py)      |          [model](https://download.openmmlab.com/mmdetection/v2.0/fpg/retinanet_r50_fpg_crop640_50e_coco/retinanet_r50_fpg_crop640_50e_coco_20220311_110809-b0bcf5f4.pth) \|                   [log](https://download.openmmlab.com/mmdetection/v2.0/fpg/retinanet_r50_fpg_crop640_50e_coco/retinanet_r50_fpg_crop640_50e_coco_20220311_110809.log.json)          |
|  RetinaNet   | FPG-chn128 |   50e   |   19.9   |       -        |  39.9  |    -    |  [config](./retinanet_r50_fpg-chn128_crop640_50e_coco.py)  |   [model](https://download.openmmlab.com/mmdetection/v2.0/fpg/retinanet_r50_fpg-chn128_crop640_50e_coco/retinanet_r50_fpg-chn128_crop640_50e_coco_20220313_104829-ee99a686.pth) \|     [log](https://download.openmmlab.com/mmdetection/v2.0/fpg/retinanet_r50_fpg-chn128_crop640_50e_coco/retinanet_r50_fpg-chn128_crop640_50e_coco_20220313_104829.log.json)   |

**Note**: Chn128 means to decrease the number of channels of features and convs from 256 (default) to 128 in
Neck and BBox Head, which can greatly decrease memory consumption without sacrificing much precision.

## Citation

```latex
@article{chen2020feature,
  title={Feature pyramid grids},
  author={Chen, Kai and Cao, Yuhang and Loy, Chen Change and Lin, Dahua and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2004.03580},
  year={2020}
}
```
