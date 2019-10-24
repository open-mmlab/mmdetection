# CARAFE: Content-Aware ReAssembly of FEatures

## Introduction

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

| Method |    Backbone     |  Style  | Lr schd | Test Proposal Num| Box AP |   Mask AP |                                                                 Download                                                                    |
| :----: | :-------------: | :-----: | :-----: | :------: | :------: |:-----------------: |:--------------------------------------------------------------------------------------------------------------------------------------------: |
| Faster R-CNN w/ CARAFE |    R-50-FPN  |  pytorch  |   1x    | 1000 |  37.8  | -  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/carafe/faster_rcnn_r50_fpn_carafe_1x-a2bf3f56.pth)  |
| - |    -  |  -  |   -    | 2000 |  37.9  | -  | -  |
| Mask R-CNN w/ CARAFE |    R-50-FPN  |  pytorch  |   1x   | 1000 |  38.6   | 35.6| [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/carafe/mask_rcnn_r50_fpn_carafe_1x-5fc51cfd.pth) |
| - |   -  |  -  |   -   | 2000 |  38.6   | 35.7| - |

## Implementation

The CUDA implementation of CARAFE can be find at `mmdet/ops/carafe` undet his repository.

