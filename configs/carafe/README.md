# CARAFE: Content-Aware ReAssembly of FEatures

## Introduction

<!-- [ALGORITHM] -->

We provide config files to reproduce the object detection & instance segmentation results in the ICCV 2019 Oral paper for [CARAFE: Content-Aware ReAssembly of FEatures](https://arxiv.org/abs/1905.02188).

```
@inproceedings{Wang_2019_ICCV,
    title = {CARAFE: Content-Aware ReAssembly of FEatures},
    author = {Wang, Jiaqi and Chen, Kai and Xu, Rui and Liu, Ziwei and Loy, Chen Change and Lin, Dahua},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2019}
}
```

## Results and Models

The results on COCO 2017 val is shown in the below table.

| Method               | Backbone | Style   | Lr schd | Test Proposal Num | Inf time (fps) | Box AP | Mask AP | Config | Download |
|:--------------------:|:--------:|:-------:|:-------:|:-----------------:|:--------------:|:------:|:-------:|:------:|:--------:|
| Faster R-CNN w/ CARAFE | R-50-FPN | pytorch | 1x      | 1000 | 16.5 | 38.6   | 38.6       | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/carafe/faster_rcnn_r50_fpn_carafe_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/carafe/faster_rcnn_r50_fpn_carafe_1x_coco/faster_rcnn_r50_fpn_carafe_1x_coco_bbox_mAP-0.386_20200504_175733-385a75b7.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/carafe/faster_rcnn_r50_fpn_carafe_1x_coco/faster_rcnn_r50_fpn_carafe_1x_coco_20200504_175733.log.json) |
| -                      |    -     |  -      | -       | 2000 |      |        |            |  |
| Mask R-CNN w/ CARAFE   | R-50-FPN | pytorch | 1x      | 1000 | 14.0 | 39.3   | 35.8       | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/carafe/mask_rcnn_r50_fpn_carafe_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/carafe/mask_rcnn_r50_fpn_carafe_1x_coco/mask_rcnn_r50_fpn_carafe_1x_coco_bbox_mAP-0.393__segm_mAP-0.358_20200503_135957-8687f195.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/carafe/mask_rcnn_r50_fpn_carafe_1x_coco/mask_rcnn_r50_fpn_carafe_1x_coco_20200503_135957.log.json) |
| -                      |   -      |  -      |   -     | 2000 |      |        |            |  |

## Implementation

The CUDA implementation of CARAFE can be find at https://github.com/myownskyW7/CARAFE.
