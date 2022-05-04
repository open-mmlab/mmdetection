# GN

> [Group Normalization](https://arxiv.org/abs/1803.08494)

<!-- [ALGORITHM] -->

## Abstract

Batch Normalization (BN) is a milestone technique in the development of deep learning, enabling various networks to train. However, normalizing along the batch dimension introduces problems --- BN's error increases rapidly when the batch size becomes smaller, caused by inaccurate batch statistics estimation. This limits BN's usage for training larger models and transferring features to computer vision tasks including detection, segmentation, and video, which require small batches constrained by memory consumption. In this paper, we present Group Normalization (GN) as a simple alternative to BN. GN divides the channels into groups and computes within each group the mean and variance for normalization. GN's computation is independent of batch sizes, and its accuracy is stable in a wide range of batch sizes. On ResNet-50 trained in ImageNet, GN has 10.6% lower error than its BN counterpart when using a batch size of 2; when using typical batch sizes, GN is comparably good with BN and outperforms other normalization variants. Moreover, GN can be naturally transferred from pre-training to fine-tuning. GN can outperform its BN-based counterparts for object detection and segmentation in COCO, and for video classification in Kinetics, showing that GN can effectively replace the powerful BN in a variety of tasks. GN can be easily implemented by a few lines of code in modern libraries.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143889171-d4660307-3a4f-4ac2-a0ad-7d17c2f045b1.png" height="200"/>
</div>

## Results and Models

| Backbone      | model      | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
|:-------------:|:----------:|:-------:|:--------:|:--------------:|:------:|:-------:|:------:|:--------:|
| R-50-FPN (d)  | Mask R-CNN | 2x      | 7.1      | 11.0           | 40.2   | 36.4    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gn/mask_rcnn_r50_fpn_gn-all_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r50_fpn_gn-all_2x_coco/mask_rcnn_r50_fpn_gn-all_2x_coco_20200206-8eee02a6.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r50_fpn_gn-all_2x_coco/mask_rcnn_r50_fpn_gn-all_2x_coco_20200206_050355.log.json) |
| R-50-FPN (d)  | Mask R-CNN | 3x      | 7.1      | -              | 40.5   | 36.7    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gn/mask_rcnn_r50_fpn_gn-all_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r50_fpn_gn-all_3x_coco/mask_rcnn_r50_fpn_gn-all_3x_coco_20200214-8b23b1e5.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r50_fpn_gn-all_3x_coco/mask_rcnn_r50_fpn_gn-all_3x_coco_20200214_063512.log.json) |
| R-101-FPN (d) | Mask R-CNN | 2x      | 9.9      | 9.0            | 41.9   | 37.6    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gn/mask_rcnn_r101_fpn_gn-all_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r101_fpn_gn-all_2x_coco/mask_rcnn_r101_fpn_gn-all_2x_coco_20200205-d96b1b50.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r101_fpn_gn-all_2x_coco/mask_rcnn_r101_fpn_gn-all_2x_coco_20200205_234402.log.json) |
| R-101-FPN (d) | Mask R-CNN | 3x      | 9.9      |                | 42.1   | 38.0    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gn/mask_rcnn_r101_fpn_gn-all_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r101_fpn_gn-all_3x_coco/mask_rcnn_r101_fpn_gn-all_3x_coco_20200513_181609-0df864f4.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r101_fpn_gn-all_3x_coco/mask_rcnn_r101_fpn_gn-all_3x_coco_20200513_181609.log.json) |
| R-50-FPN (c)  | Mask R-CNN | 2x      | 7.1      | 10.9           | 40.0   | 36.1    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gn/mask_rcnn_r50_fpn_gn-all_contrib_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r50_fpn_gn-all_contrib_2x_coco/mask_rcnn_r50_fpn_gn-all_contrib_2x_coco_20200207-20d3e849.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r50_fpn_gn-all_contrib_2x_coco/mask_rcnn_r50_fpn_gn-all_contrib_2x_coco_20200207_225832.log.json) |
| R-50-FPN (c)  | Mask R-CNN | 3x      | 7.1      | -              | 40.1   | 36.2    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gn/mask_rcnn_r50_fpn_gn-all_contrib_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r50_fpn_gn-all_contrib_3x_coco/mask_rcnn_r50_fpn_gn-all_contrib_3x_coco_20200225-542aefbc.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r50_fpn_gn-all_contrib_3x_coco/mask_rcnn_r50_fpn_gn-all_contrib_3x_coco_20200225_235135.log.json) |

**Notes:**

- (d) means pretrained model converted from Detectron, and (c) means the contributed model pretrained by [@thangvubk](https://github.com/thangvubk).
- The `3x` schedule is epoch [28, 34, 36].
- **Memory, Train/Inf time is outdated.**

## Citation

```latex
@inproceedings{wu2018group,
  title={Group Normalization},
  author={Wu, Yuxin and He, Kaiming},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2018}
}
```
