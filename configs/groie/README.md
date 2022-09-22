# GRoIE

> [A novel Region of Interest Extraction Layer for Instance Segmentation](https://arxiv.org/abs/2004.13665)

<!-- [ALGORITHM] -->

## Abstract

Given the wide diffusion of deep neural network architectures for computer vision tasks, several new applications are nowadays more and more feasible. Among them, a particular attention has been recently given to instance segmentation, by exploiting the results achievable by two-stage networks (such as Mask R-CNN or Faster R-CNN), derived from R-CNN. In these complex architectures, a crucial role is played by the Region of Interest (RoI) extraction layer, devoted to extracting a coherent subset of features from a single Feature Pyramid Network (FPN) layer attached on top of a backbone.
This paper is motivated by the need to overcome the limitations of existing RoI extractors which select only one (the best) layer from FPN. Our intuition is that all the layers of FPN retain useful information. Therefore, the proposed layer (called Generic RoI Extractor - GRoIE) introduces non-local building blocks and attention mechanisms to boost the performance.
A comprehensive ablation study at component level is conducted to find the best set of algorithms and parameters for the GRoIE layer. Moreover, GRoIE can be integrated seamlessly with every two-stage architecture for both object detection and instance segmentation tasks. Therefore, the improvements brought about by the use of GRoIE in different state-of-the-art architectures are also evaluated. The proposed layer leads up to gain a 1.1% AP improvement on bounding box detection and 1.7% AP improvement on instance segmentation.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143891453-afdcdaf4-a868-4a28-ad20-dc710a517a76.png"/>
</div>

## Introduction

By Leonardo Rossi, Akbar Karimi and Andrea Prati from
[IMPLab](http://implab.ce.unipr.it/).

We provide configs to reproduce the results in the paper for
"*A novel Region of Interest Extraction Layer for Instance Segmentation*"
on COCO object detection.

This paper is motivated by the need to overcome to the limitations of existing
RoI extractors which select only one (the best) layer from FPN.

Our intuition is that all the layers of FPN retain useful information.

Therefore, the proposed layer (called Generic RoI Extractor - **GRoIE**)
introduces non-local building blocks and attention mechanisms to boost the
performance.

## Results and Models

The results on COCO 2017 minival (5k images) are shown in the below table.

### Application of GRoIE to different architectures

| Backbone  |     Method      | Lr schd | box AP | mask AP |                                Config                                |                                                                                                                                                                                                                   Download                                                                                                                                                                                                                   |
| :-------: | :-------------: | :-----: | :----: | :-----: | :------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| R-50-FPN  | Faster Original |   1x    |  37.4  |         |       [config](../faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py)        |                                                                  [model](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130_204655.log.json)                                                                   |
| R-50-FPN  |     + GRoIE     |   1x    |  38.3  |         |           [config](./faste-rcnn_r50_fpn_groie_1x_coco.py)            |                                                         [model](https://download.openmmlab.com/mmdetection/v2.0/groie/faster_rcnn_r50_fpn_groie_1x_coco/faster_rcnn_r50_fpn_groie_1x_coco_20200604_211715-66ee9516.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/groie/faster_rcnn_r50_fpn_groie_1x_coco/faster_rcnn_r50_fpn_groie_1x_coco_20200604_211715.log.json)                                                         |
| R-50-FPN  |   Grid R-CNN    |   1x    |  39.1  |         |        [config](./grid-rcnn_r50_fpn_gn-head-groie_1x_coco.py)        |                                             [model](https://download.openmmlab.com/mmdetection/v2.0/groie/grid_rcnn_r50_fpn_gn-head_groie_1x_coco/grid_rcnn_r50_fpn_gn-head_groie_1x_coco_20200605_202059-4b75d86f.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/groie/grid_rcnn_r50_fpn_gn-head_groie_1x_coco/grid_rcnn_r50_fpn_gn-head_groie_1x_coco_20200605_202059.log.json)                                             |
| R-50-FPN  |     + GRoIE     |   1x    |        |         |        [config](./grid-rcnn_r50_fpn_gn-head-groie_1x_coco.py)        |                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| R-50-FPN  |   Mask R-CNN    |   1x    |  38.2  |  34.7   |         [config](../mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py)          |                                                                        [model](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205_050542.log.json)                                                                         |
| R-50-FPN  |     + GRoIE     |   1x    |  39.0  |  36.0   |            [config](./mask-rcnn_r50_fpn_groie_1x_coco.py)            |                                                             [model](https://download.openmmlab.com/mmdetection/v2.0/groie/mask_rcnn_r50_fpn_groie_1x_coco/mask_rcnn_r50_fpn_groie_1x_coco_20200604_211715-50d90c74.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/groie/mask_rcnn_r50_fpn_groie_1x_coco/mask_rcnn_r50_fpn_groie_1x_coco_20200604_211715.log.json)                                                             |
| R-50-FPN  |     GC-Net      |   1x    |  40.7  |  36.5   | [config](../gcnet/mask-rcnn_r50-syncbn-gcb-r4-c3-c5_fpn_1x_coco.py)  |                  [model](https://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200202-50b90e5c.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200202_085547.log.json)                   |
| R-50-FPN  |     + GRoIE     |   1x    |  41.0  |  37.8   |  [config](./mask-rcnn_r50_fpn_syncbn-r4-gcb-c3-c5-groie_1x_coco.py)  | [model](https://download.openmmlab.com/mmdetection/v2.0/groie/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco_20200604_211715-42eb79e1.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/groie/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco_20200604_211715-42eb79e1.pth) |
| R-101-FPN |     GC-Net      |   1x    |  42.2  |  37.8   | [config](../gcnet/mask-rcnn_r101-syncbn-gcb-r4-c3-c5_fpn_1x_coco.py) |                [model](https://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200206-8407a3f0.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200206_142508.log.json)                 |
| R-101-FPN |     + GRoIE     |   1x    |  42.6  |  38.7   | [config](./mask-rcnn_r101_fpn_syncbn-r4-gcb_c3-c5-groie_1x_coco.py)  | [model](https://download.openmmlab.com/mmdetection/v2.0/groie/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco_20200607_224507-8daae01c.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/groie/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco_20200607_224507.log.json) |

## Citation

If you use this work or benchmark in your research, please cite this project.

```latex
@inproceedings{rossi2021novel,
  title={A novel region of interest extraction layer for instance segmentation},
  author={Rossi, Leonardo and Karimi, Akbar and Prati, Andrea},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
  pages={2203--2209},
  year={2021},
  organization={IEEE}
}
```

## Contact

The implementation of GRoIE is currently maintained by
[Leonardo Rossi](https://github.com/hachreak/).
