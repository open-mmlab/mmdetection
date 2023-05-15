# ConsistentTeacher

> [Consistent-Teacher: Towards Reducing Inconsistent Pseudo-targets in Semi-supervised Object Detection](https://arxiv.org/abs/2209.01589)

<!-- [ALGORITHM] -->

## Abstract

In this study, we dive deep into the inconsistency of pseudo targets in semi-supervised object detection (SSOD). Our core observation is that the oscillating pseudo-targets undermine the training of an accurate detector. It injects noise into the student's training, leading to severe overfitting problems. Therefore, we propose a systematic solution, termed ConsistentTeacher, to reduce the inconsistency. First, adaptive anchor assignment~(ASA) substitutes the static IoU-based strategy, which enables the student network to be resistant to noisy pseudo-bounding boxes. Then we calibrate the subtask predictions by designing a 3D feature alignment module~(FAM-3D). It allows each classification feature to adaptively query the optimal feature vector for the regression task at arbitrary scales and locations. Lastly, a Gaussian Mixture Model (GMM) dynamically revises the score threshold of pseudo-bboxes, which stabilizes the number of ground truths at an early stage and remedies the unreliable supervision signal during training. ConsistentTeacher provides strong results on a large range of SSOD evaluations. It achieves 40.0 mAP with ResNet-50 backbone given only 10% of annotated MS-COCO data, which surpasses previous baselines using pseudo labels by around 3 mAP. When trained on fully annotated MS-COCO with additional unlabeled data, the performance further increases to 47.7 mAP.

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/7219519/6c80c611-74dc-486a-ac84-a590ea28e258"/>
</div>

## Results and Models

|       Model       | Detector  | Labeled Dataset | Iteration | box AP |                                 Config                                  |         Download         |
| :---------------: | :-------: | :-------------: | :-------: | :----: | :---------------------------------------------------------------------: | :----------------------: |
| ConsistentTeacher | RetinaNet |    COCO-10%     |   180k    |        | [config](./consistent-teacher_retinanet_r50_fpn_180k_semi-0.1-coco.py)  | [model](<>) \| [log](<>) |
| ConsistentTeacher | RetinaNet |     COCO-1%     |   180k    |        | [config](./consistent-teacher_retinanet_r50_fpn_180k_semi-0.01-coco.py) | [model](<>) \| [log](<>) |
| ConsistentTeacher | RetinaNet |     COCO-2%     |   180k    |        | [config](./consistent-teacher_retinanet_r50_fpn_180k_semi-0.02-coco.py) | [model](<>) \| [log](<>) |
| ConsistentTeacher | RetinaNet |     COCO-5%     |   180k    |        | [config](./consistent-teacher_retinanet_r50_fpn_180k_semi-0.05-coco.py) | [model](<>) \| [log](<>) |
| ConsistentTeacher | RetinaNet |    COCO-100%    |   720k    |        |   [config](./consistent-teacher_retinanet_r50_fpn_720k_semi-coco.py)    | [model](<>) \| [log](<>) |
| ConsistentTeacher | RetinaNet |     VOC0712     |    72k    |        |  [config](./consistent-teacher_retinanet_r50_fpn_72k_semi-voc0712.py)   | [model](<>) \| [log](<>) |

## Citation

```latex
@article{wang2023consistent,
    author    = {Xinjiang Wang, Xingyi Yang, Shilong Zhang, Yijiang Li, Litong Feng, Shijie Fang, Chengqi Lyu, Kai Chen, Wayne Zhang },
    title     = {Consistent-Teacher: Towards Reducing Inconsistent Pseudo-targets in Semi-supervised Object Detection},
    journal   = {The IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR)},
    year      = {2023},
}
```
