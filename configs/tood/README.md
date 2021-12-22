# TOOD: Task-aligned One-stage Object Detection

## Abstract

<!-- [ABSTRACT] -->

One-stage object detection is commonly implemented by optimizing two sub-tasks: object classification and localization, using heads with two parallel branches, which might lead to a certain level of spatial misalignment in predictions between the two tasks. In this work, we propose a Task-aligned One-stage Object Detection (TOOD) that explicitly aligns the two tasks in a learning-based manner. First, we design a novel Task-aligned Head (T-Head) which offers a better balance between learning task-interactive and task-specific features, as well as a greater flexibility to learn the alignment via a task-aligned predictor. Second, we propose Task Alignment Learning (TAL) to explicitly pull closer (or even unify) the optimal anchors for the two tasks during training via a designed sample assignment scheme and a task-aligned loss. Extensive experiments are conducted on MS-COCO, where TOOD achieves a 51.1 AP at single-model single-scale testing. This surpasses the recent one-stage detectors by a large margin, such as ATSS (47.7 AP), GFL (48.2 AP), and PAA (49.0 AP), with fewer parameters and FLOPs. Qualitative results also demonstrate the effectiveness of TOOD for better aligning the tasks of object classification and localization.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/12907710/145400075-e08191f5-8afa-4335-9b3b-27926fc9a26e.png"/>
</div>

<!-- [PAPER_TITLE: TOOD: Task-aligned One-stage Object Detection] -->
<!-- [PAPER_URL: https://arxiv.org/abs/2108.07755] -->

## Citation

<!-- [ALGORITHM] -->

```latex
@inproceedings{feng2021tood,
    title={TOOD: Task-aligned One-stage Object Detection},
    author={Feng, Chengjian and Zhong, Yujie and Gao, Yu and Scott, Matthew R and Huang, Weilin},
    booktitle={ICCV},
    year={2021}
}
```

## Results and Models

| Backbone          | Style   | Anchor Type  | Lr schd | Multi-scale Training| Mem (GB)| Inf time (fps) | box AP | Config | Download |
|:-----------------:|:-------:|:------------:|:-------:|:-------------------:|:-------:|:--------------:|:------:|:------:|:--------:|
| R-50              | pytorch | Anchor-free  | 1x      | N                   | 4.1     |                | 42.4   | [config](./tood_r50_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/tood/tood_r50_fpn_1x_coco/tood_r50_fpn_1x_coco_20211210_103425-20e20746.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/tood/tood_r50_fpn_1x_coco/tood_r50_fpn_1x_coco_20211210_103425.log) |
| R-50              | pytorch | Anchor-based | 1x      | N                   | 4.1     |                | 42.4   | [config](./tood_r50_fpn_anchor_based_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/tood/tood_r50_fpn_anchor_based_1x_coco/tood_r50_fpn_anchor_based_1x_coco_20211214_100105-b776c134.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/tood/tood_r50_fpn_anchor_based_1x_coco/tood_r50_fpn_anchor_based_1x_coco_20211214_100105.log) |
| R-50              | pytorch | Anchor-free  | 2x      | Y                   | 4.1     |                | 44.5   | [config](./tood_r50_fpn_mstrain_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/tood/tood_r50_fpn_mstrain_2x_coco/tood_r50_fpn_mstrain_2x_coco_20211210_144231-3b23174c.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/tood/tood_r50_fpn_mstrain_2x_coco/tood_r50_fpn_mstrain_2x_coco_20211210_144231.log) |
| R-101             | pytorch | Anchor-free  | 2x      | Y                   | 6.0     |                | 46.1   | [config](./tood_r101_fpn_mstrain_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/tood/tood_r101_fpn_mstrain_2x_coco/tood_r101_fpn_mstrain_2x_coco_20211210_144232-a18f53c8.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/tood/tood_r101_fpn_mstrain_2x_coco/tood_r101_fpn_mstrain_2x_coco_20211210_144232.log) |
| R-101-dcnv2       | pytorch | Anchor-free  | 2x      | Y                   | 6.2     |                | 49.3   | [config](./tood_r101_fpn_dconv_c3-c5_mstrain_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/tood/tood_r101_fpn_dconv_c3-c5_mstrain_2x_coco/tood_r101_fpn_dconv_c3-c5_mstrain_2x_coco_20211210_213728-4a824142.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/tood/tood_r101_fpn_dconv_c3-c5_mstrain_2x_coco/tood_r101_fpn_dconv_c3-c5_mstrain_2x_coco_20211210_213728.log) |
| X-101-64x4d       | pytorch | Anchor-free  | 2x      | Y                   | 10.2    |                | 47.6   | [config](./tood_x101_64x4d_fpn_mstrain_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/tood/tood_x101_64x4d_fpn_mstrain_2x_coco/tood_x101_64x4d_fpn_mstrain_2x_coco_20211211_003519-a4f36113.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/tood/tood_x101_64x4d_fpn_mstrain_2x_coco/tood_x101_64x4d_fpn_mstrain_2x_coco_20211211_003519.log) |
| X-101-64x4d-dcnv2 | pytorch | Anchor-free  | 2x      | Y                   |         |                |        | [config](./tood_x101_64x4d_fpn_dconv_c4-c5_mstrain_2x_coco.py) | [model]() &#124; [log]() |

[1] *1x and 2x mean the model is trained for 90K and 180K iterations, respectively.* \
[2] *All results are obtained with a single model and without any test time data augmentation such as multi-scale, flipping and etc..* \
[3] *`dcnv2` denotes deformable convolutional networks v2.* \
