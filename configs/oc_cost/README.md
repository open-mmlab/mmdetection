# OC Cost

> [Optimal Correction Cost for Object Detection Evaluation](https://arxiv.org/abs/2203.14438)

<!-- [DATASET] -->

## Abstract

Mean Average Precision (mAP) is the primary evaluation measure for object detection. Although object detection has a broad range of applications, mAP evaluates detectors in terms of the performance of ranked instance retrieval. Such the assumption for the evaluation task does not suit some downstream tasks. To alleviate the gap between downstream tasks and the evaluation scenario, we propose Optimal Correction Cost (OC-cost), which assesses detection accuracy at image level. OC-cost computes the cost of correcting detections to ground truths as a measure of accuracy. The cost is obtained by solving an optimal transportation problem between the detections and the ground truths. Unlike mAP, OC-cost is designed to penalize false positive and false negative detections properly, and every image in a dataset is treated equally. Our experimental result validates that OC-cost has better agreement with human preference than a ranking-based measure, i.e., mAP for a single image. We also show that detectors' rankings by OC-cost are more consistent on different data splits than mAP. Our goal is not to replace mAP with OC-cost but provide an additional tool to evaluate detectors from another aspect. To help future researchers and developers choose a target measure, we provide a series of experiments to clarify how mAP and OC-cost differ.

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/24734142/8a04fbfb-1fac-4514-8dd5-6546a1f9a04c" height="300"/>
</div>

## Results and models

| Architecture | Backbone | box OC Cost |                       Config                       |                                                                      Download                                                                      |
| :----------: | :------: | :---------: | :------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------: |
| Faster R-CNN |   R-50   |   0.5563    | [config](./faster-rcnn_r50_fpn_1x_coco_oc_cost.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth) |
|  Retinanet   |   R-50   |   0.5819    |  [config](./retinanet_r50_fpn_1x_coco_oc_cost.py)  |    [model](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth)    |

## Citation

```latex
@InProceedings{Otani_2022_CVPR,
    author    = {Otani, Mayu and Togashi, Riku and Nakashima, Yuta and Rahtu, Esa and Heikkil\"a, Janne and Satoh, Shin'ichi},
    title     = {Optimal Correction Cost for Object Detection Evaluation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {21107-21115}
}
```
