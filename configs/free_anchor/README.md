# FreeAnchor

> [FreeAnchor: Learning to Match Anchors for Visual Object Detection](https://arxiv.org/abs/1909.02466)

<!-- [ALGORITHM] -->

## Abstract

Modern CNN-based object detectors assign anchors for ground-truth objects under the restriction of object-anchor Intersection-over-Unit (IoU). In this study, we propose a learning-to-match approach to break IoU restriction, allowing objects to match anchors in a flexible manner. Our approach, referred to as FreeAnchor, updates hand-crafted anchor assignment to "free" anchor matching by formulating detector training as a maximum likelihood estimation (MLE) procedure. FreeAnchor targets at learning features which best explain a class of objects in terms of both classification and localization. FreeAnchor is implemented by optimizing detection customized likelihood and can be fused with CNN-based detectors in a plug-and-play manner. Experiments on COCO demonstrate that FreeAnchor consistently outperforms their counterparts with significant margins.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143886006-8374bf9b-28af-442e-9abf-eb16562bb7d3.png"/>
</div>

## Results and Models

|  Backbone   |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP |                      Config                      |                                                                                                                                                                                   Download                                                                                                                                                                                    |
| :---------: | :-----: | :-----: | :------: | :------------: | :----: | :----------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|    R-50     | pytorch |   1x    |   4.9    |      18.4      |  38.7  |    [config](./freeanchor_r50_fpn_1x_coco.py)     |               [model](https://download.openmmlab.com/mmdetection/v2.0/free_anchor/retinanet_free_anchor_r50_fpn_1x_coco/retinanet_free_anchor_r50_fpn_1x_coco_20200130-0f67375f.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/free_anchor/retinanet_free_anchor_r50_fpn_1x_coco/retinanet_free_anchor_r50_fpn_1x_coco_20200130_095625.log.json)               |
|    R-101    | pytorch |   1x    |   6.8    |      14.9      |  40.3  |    [config](./freeanchor_r101_fpn_1x_coco.py)    |             [model](https://download.openmmlab.com/mmdetection/v2.0/free_anchor/retinanet_free_anchor_r101_fpn_1x_coco/retinanet_free_anchor_r101_fpn_1x_coco_20200130-358324e6.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/free_anchor/retinanet_free_anchor_r101_fpn_1x_coco/retinanet_free_anchor_r101_fpn_1x_coco_20200130_100723.log.json)             |
| X-101-32x4d | pytorch |   1x    |   8.1    |      11.1      |  41.9  | [config](./freeanchor_x101-32x4d_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/free_anchor/retinanet_free_anchor_x101_32x4d_fpn_1x_coco/retinanet_free_anchor_x101_32x4d_fpn_1x_coco_20200130-d4846968.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/free_anchor/retinanet_free_anchor_x101_32x4d_fpn_1x_coco/retinanet_free_anchor_x101_32x4d_fpn_1x_coco_20200130_095627.log.json) |

**Notes:**

- We use 8 GPUs with 2 images/GPU.
- For more settings and models, please refer to the [official repo](https://github.com/zhangxiaosong18/FreeAnchor).

## Citation

```latex
@inproceedings{zhang2019freeanchor,
  title   =  {{FreeAnchor}: Learning to Match Anchors for Visual Object Detection},
  author  =  {Zhang, Xiaosong and Wan, Fang and Liu, Chang and Ji, Rongrong and Ye, Qixiang},
  booktitle =  {Neural Information Processing Systems},
  year    =  {2019}
}
```
