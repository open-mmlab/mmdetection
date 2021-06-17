# Deformable DETR

## Introduction

<!-- [ALGORITHM] -->

We provide the config files for Deformable DETR: [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159).

```
@inproceedings{
zhu2021deformable,
title={Deformable DETR: Deformable Transformers for End-to-End Object Detection},
author={Xizhou Zhu and Weijie Su and Lewei Lu and Bin Li and Xiaogang Wang and Jifeng Dai},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=gZ9hCDWe6ke}
}
```

## Results and Models

| Backbone | Model | Lr schd  | box AP | Config | Download |
|:------:|:--------:|:--------------:|:------:|:------:|:--------:|
| R-50 | Deformable DETR  |50e  | 44.5 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_r50_16x2_50e_coco/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_r50_16x2_50e_coco/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.log.json) |
| R-50 | + iterative bounding box refinement  |50e  | 46.1 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/deformable_detr/deformable_detr_refine_r50_16x2_50e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_refine_r50_16x2_50e_coco/deformable_detr_refine_r50_16x2_50e_coco_20210419_220503-5f5dff21.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_refine_r50_16x2_50e_coco/deformable_detr_refine_r50_16x2_50e_coco_20210419_220503-5f5dff21.log.json) |
| R-50 | ++ two-stage Deformable DETR  |50e  | 46.8 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.log.json) |

# NOTE

1. All models are trained with batch size 32.
2. The performance is unstable. `Deformable DETR` and `iterative bounding box refinement` may fluctuate about 0.3 mAP. `two-stage Deformable DETR` may fluctuate about 0.2 mAP.
