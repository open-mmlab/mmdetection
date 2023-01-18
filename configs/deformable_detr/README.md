# Deformable DETR

> [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159)

<!-- [ALGORITHM] -->

## Abstract

DETR has been recently proposed to eliminate the need for many hand-designed components in object detection while demonstrating good performance. However, it suffers from slow convergence and limited feature spatial resolution, due to the limitation of Transformer attention modules in processing image feature maps. To mitigate these issues, we proposed Deformable DETR, whose attention modules only attend to a small set of key sampling points around a reference. Deformable DETR can achieve better performance than DETR (especially on small objects) with 10 times less training epochs. Extensive experiments on the COCO benchmark demonstrate the effectiveness of our approach.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143877617-ad9b24fd-77ce-46aa-9689-1a44b5594132.png"/>
</div>

## Results and Models

| Backbone |                Model                | Lr schd | box AP |                              Config                               |                                                                                                                                                                                                       Download                                                                                                                                                                                                       |
| :------: | :---------------------------------: | :-----: | :----: | :---------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-50   |           Deformable DETR           |   50e   |  44.3  |         [config](./deformable-detr_r50_16xb2-50e_coco.py)         |                                 [model](https://download.openmmlab.com/mmdetection/v3.0/deformable_detr/deformable-detr_r50_16xb2-50e_coco/deformable-detr_r50_16xb2-50e_coco_20221029_210934-6bc7d21b.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/deformable_detr/deformable-detr_r50_16xb2-50e_coco/deformable-detr_r50_16xb2-50e_coco_20221029_210934.log.json)                                 |
|   R-50   | + iterative bounding box refinement |   50e   |  46.2  |     [config](./deformable-detr-refine_r50_16xb2-50e_coco.py)      |                   [model](https://download.openmmlab.com/mmdetection/v3.0/deformable_detr/deformable-detr-refine_r50_16xb2-50e_coco/deformable-detr-refine_r50_16xb2-50e_coco_20221022_225303-844e0f93.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/deformable_detr/deformable-detr-refine_r50_16xb2-50e_coco/deformable-detr-refine_r50_16xb2-50e_coco_20221022_225303.log.json)                   |
|   R-50   |    ++ two-stage Deformable DETR     |   50e   |  47.0  | [config](./deformable-detr-refine-twostage_r50_16xb2-50e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/deformable_detr/deformable-detr-refine-twostage_r50_16xb2-50e_coco/deformable-detr-refine-twostage_r50_16xb2-50e_coco_20221021_184714-acc8a5ff.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/deformable_detr/deformable-detr-refine-twostage_r50_16xb2-50e_coco/deformable-detr-refine-twostage_r50_16xb2-50e_coco_20221021_184714.log.json) |

### NOTE

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
