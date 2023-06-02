# DETR

> [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

<!-- [ALGORITHM] -->

## Abstract

We present a new method that views object detection as a direct set prediction problem. Our approach streamlines the detection pipeline, effectively removing the need for many hand-designed components like a non-maximum suppression procedure or anchor generation that explicitly encode our prior knowledge about the task. The main ingredients of the new framework, called DEtection TRansformer or DETR, are a set-based global loss that forces unique predictions via bipartite matching, and a transformer encoder-decoder architecture. Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel. The new model is conceptually simple and does not require a specialized library, unlike many other modern detectors. DETR demonstrates accuracy and run-time performance on par with the well-established and highly-optimized Faster RCNN baseline on the challenging COCO object detection dataset. Moreover, DETR can be easily generalized to produce panoptic segmentation in a unified manner. We show that it significantly outperforms competitive baselines.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143878072-0a7434e4-416b-4315-aeea-a8297f4d6453.png"/>
</div>

## Results and Models

| Backbone | Model | Lr schd | Mem (GB) | Inf time (fps) | box AP |                                                 Config                                                 |                                                                                                                                    Download                                                                                                                                    |
| :------: | :---: | :-----: | :------: | :------------: | :----: | :----------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-50   | DETR  |  150e   |   7.9    |                |  40.1  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/detr/detr_r50_8x2_150e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835.log.json) |

## Citation

We provide the config files for DETR: [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872).

```latex
@inproceedings{detr,
  author    = {Nicolas Carion and
               Francisco Massa and
               Gabriel Synnaeve and
               Nicolas Usunier and
               Alexander Kirillov and
               Sergey Zagoruyko},
  title     = {End-to-End Object Detection with Transformers},
  booktitle = {ECCV},
  year      = {2020}
}
```
