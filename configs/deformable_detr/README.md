# Deformable DETR

> [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159)

<!-- [ALGORITHM] -->

## Abstract

DETR has been recently proposed to eliminate the need for many hand-designed components in object detection while demonstrating good performance. However, it suffers from slow convergence and limited feature spatial resolution, due to the limitation of Transformer attention modules in processing image feature maps. To mitigate these issues, we proposed Deformable DETR, whose attention modules only attend to a small set of key sampling points around a reference. Deformable DETR can achieve better performance than DETR (especially on small objects) with 10 times less training epochs. Extensive experiments on the COCO benchmark demonstrate the effectiveness of our approach.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143877617-ad9b24fd-77ce-46aa-9689-1a44b5594132.png"/>
</div>

## Results and Models

| Backbone |                Model                | Lr schd | box AP |                                                                    Config                                                                    |                                                                                                                                                                                                         Download                                                                                                                                                                                                          |
| :------: | :---------------------------------: | :-----: | :----: | :------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-50   |           Deformable DETR           |   50e   |  44.5  |         [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py)         |                                 [model](https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_r50_16x2_50e_coco/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_r50_16x2_50e_coco/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.log.json)                                 |
|   R-50   | + iterative bounding box refinement |   50e   |  46.1  |     [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/deformable_detr/deformable_detr_refine_r50_16x2_50e_coco.py)      |                   [model](https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_refine_r50_16x2_50e_coco/deformable_detr_refine_r50_16x2_50e_coco_20210419_220503-5f5dff21.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_refine_r50_16x2_50e_coco/deformable_detr_refine_r50_16x2_50e_coco_20210419_220503-5f5dff21.log.json)                   |
|   R-50   |    ++ two-stage Deformable DETR     |   50e   |  46.8  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.log.json) |

# NOTE

1. All models are trained with batch size 32.
2. The performance is unstable. `Deformable DETR` and `iterative bounding box refinement` may fluctuate about 0.3 mAP. `two-stage Deformable DETR` may fluctuate about 0.2 mAP.

## Citation

We provide the config files for Deformable DETR: [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159).

```latex
@inproceedings{
zhu2021deformable,
title={Deformable DETR: Deformable Transformers for End-to-End Object Detection},
author={Xizhou Zhu and Weijie Su and Lewei Lu and Bin Li and Xiaogang Wang and Jifeng Dai},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=gZ9hCDWe6ke}
}
```
