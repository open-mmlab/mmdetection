# PAA

> [Probabilistic Anchor Assignment with IoU Prediction for Object Detection](https://arxiv.org/abs/2007.08103)

<!-- [ALGORITHM] -->

## Abstract

In object detection, determining which anchors to assign as positive or negative samples, known as anchor assignment, has been revealed as a core procedure that can significantly affect a model's performance. In this paper we propose a novel anchor assignment strategy that adaptively separates anchors into positive and negative samples for a ground truth bounding box according to the model's learning status such that it is able to reason about the separation in a probabilistic manner. To do so we first calculate the scores of anchors conditioned on the model and fit a probability distribution to these scores. The model is then trained with anchors separated into positive and negative samples according to their probabilities. Moreover, we investigate the gap between the training and testing objectives and propose to predict the Intersection-over-Unions of detected boxes as a measure of localization quality to reduce the discrepancy. The combined score of classification and localization qualities serving as a box selection metric in non-maximum suppression well aligns with the proposed anchor assignment strategy and leads significant performance improvements. The proposed methods only add a single convolutional layer to RetinaNet baseline and does not require multiple anchors per location, so are efficient. Experimental results verify the effectiveness of the proposed methods. Especially, our models set new records for single-stage detectors on MS COCO test-dev dataset with various backbones.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143968195-519a116a-de29-437e-b4c8-30aef43dcb15.png"/>
</div>

## Results and Models

We provide config files to reproduce the object detection results in the
ECCV 2020 paper for Probabilistic Anchor Assignment with IoU
Prediction for Object Detection.

| Backbone  | Lr schd | Mem (GB) | Score voting | box AP |                 Config                 |                                                                                                                                               Download                                                                                                                                               |
| :-------: | :-----: | :------: | :----------: | :----: | :------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| R-50-FPN  |   12e   |   3.7    |     True     |  40.4  |   [config](./paa_r50_fpn_1x_coco.py)   |                     [model](https://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_1x_coco/paa_r50_fpn_1x_coco_20200821-936edec3.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_1x_coco/paa_r50_fpn_1x_coco_20200821-936edec3.log.json)                      |
| R-50-FPN  |   12e   |   3.7    |    False     |  40.2  |                   -                    |                                                                                                                                                                                                                                                                                                      |
| R-50-FPN  |   18e   |   3.7    |     True     |  41.4  |  [config](./paa_r50_fpn_1.5x_coco.py)  |                 [model](https://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_1.5x_coco/paa_r50_fpn_1.5x_coco_20200823-805d6078.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_1.5x_coco/paa_r50_fpn_1.5x_coco_20200823-805d6078.log.json)                  |
| R-50-FPN  |   18e   |   3.7    |    False     |  41.2  |                   -                    |                                                                                                                                                                                                                                                                                                      |
| R-50-FPN  |   24e   |   3.7    |     True     |  41.6  |   [config](./paa_r50_fpn_2x_coco.py)   |                     [model](https://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_2x_coco/paa_r50_fpn_2x_coco_20200821-c98bfc4e.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_2x_coco/paa_r50_fpn_2x_coco_20200821-c98bfc4e.log.json)                      |
| R-50-FPN  |   36e   |   3.7    |     True     |  43.3  | [config](./paa_r50_fpn_ms-3x_coco.py)  |   [model](https://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_mstrain_3x_coco/paa_r50_fpn_mstrain_3x_coco_20210121_145722-06a6880b.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_mstrain_3x_coco/paa_r50_fpn_mstrain_3x_coco_20210121_145722.log.json)   |
| R-101-FPN |   12e   |   6.2    |     True     |  42.6  |  [config](./paa_r101_fpn_1x_coco.py)   |                   [model](https://download.openmmlab.com/mmdetection/v2.0/paa/paa_r101_fpn_1x_coco/paa_r101_fpn_1x_coco_20200821-0a1825a4.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/paa/paa_r101_fpn_1x_coco/paa_r101_fpn_1x_coco_20200821-0a1825a4.log.json)                    |
| R-101-FPN |   12e   |   6.2    |    False     |  42.4  |                   -                    |                                                                                                                                                                                                                                                                                                      |
| R-101-FPN |   24e   |   6.2    |     True     |  43.5  |  [config](./paa_r101_fpn_2x_coco.py)   |                   [model](https://download.openmmlab.com/mmdetection/v2.0/paa/paa_r101_fpn_2x_coco/paa_r101_fpn_2x_coco_20200821-6829f96b.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/paa/paa_r101_fpn_2x_coco/paa_r101_fpn_2x_coco_20200821-6829f96b.log.json)                    |
| R-101-FPN |   36e   |   6.2    |     True     |  45.1  | [config](./paa_r101_fpn_ms-3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/paa/paa_r101_fpn_mstrain_3x_coco/paa_r101_fpn_mstrain_3x_coco_20210122_084202-83250d22.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/paa/paa_r101_fpn_mstrain_3x_coco/paa_r101_fpn_mstrain_3x_coco_20210122_084202.log.json) |

**Note**:

1. We find that the performance is unstable with 1x setting and may fluctuate by about 0.2 mAP. We report the best results.

## Citation

```latex
@inproceedings{paa-eccv2020,
  title={Probabilistic Anchor Assignment with IoU Prediction for Object Detection},
  author={Kim, Kang and Lee, Hee Seok},
  booktitle = {ECCV},
  year={2020}
}
```
