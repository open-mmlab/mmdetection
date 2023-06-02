# LAD

> [Improving Object Detection by Label Assignment Distillation](https://arxiv.org/abs/2108.10520)

<!-- [ALGORITHM] -->

## Abstract

Label assignment in object detection aims to assign targets, foreground or background, to sampled regions in an image. Unlike labeling for image classification, this problem is not well defined due to the object's bounding box. In this paper, we investigate the problem from a perspective of distillation, hence we call Label Assignment Distillation (LAD). Our initial motivation is very simple, we use a teacher network to generate labels for the student. This can be achieved in two ways: either using the teacher's prediction as the direct targets (soft label), or through the hard labels dynamically assigned by the teacher (LAD). Our experiments reveal that: (i) LAD is more effective than soft-label, but they are complementary. (ii) Using LAD, a smaller teacher can also improve a larger student significantly, while soft-label can't. We then introduce Co-learning LAD, in which two networks simultaneously learn from scratch and the role of teacher and student are dynamically interchanged. Using PAA-ResNet50 as a teacher, our LAD techniques can improve detectors PAA-ResNet101 and PAA-ResNeXt101 to 46AP and 47.5AP on the COCO test-dev set. With a stronger teacher PAA-SwinB, we improve the students PAA-ResNet50 to 43.7AP by only 1x schedule training and standard setting, and PAA-ResNet101 to 47.9AP, significantly surpassing the current methods.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143894499-c2a3a243-988f-4604-915b-17918732bf03.png"/>
</div>

## Results and Models

We provide config files to reproduce the object detection results in the
WACV 2022 paper for Improving Object Detection by Label Assignment
Distillation.

### PAA with LAD

| Teacher | Student | Training schedule | AP (val) |                                                   Config                                                    |                                                                                                                                               Download                                                                                                                                               |
| :-----: | :-----: | :---------------: | :------: | :---------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   --    |  R-50   |        1x         |   40.4   |     [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/paa/paa_r50_fpn_1x_coco.py)      |                     [model](https://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_1x_coco/paa_r50_fpn_1x_coco_20200821-936edec3.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_1x_coco/paa_r50_fpn_1x_coco_20200821-936edec3.log.json)                      |
|   --    |  R-101  |        1x         |   42.6   |     [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/paa/paa_r101_fpn_1x_coco.py)     |                   [model](https://download.openmmlab.com/mmdetection/v2.0/paa/paa_r101_fpn_1x_coco/paa_r101_fpn_1x_coco_20200821-0a1825a4.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/paa/paa_r101_fpn_1x_coco/paa_r101_fpn_1x_coco_20200821-0a1825a4.log.json)                    |
|  R-101  |  R-50   |        1x         |   41.4   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/lad/lad_r50_paa_r101_fpn_coco_1x.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/lad/lad_r50_paa_r101_fpn_coco_1x/lad_r50_paa_r101_fpn_coco_1x_20220708_124246-74c76ff0.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/lad/lad_r50_paa_r101_fpn_coco_1x/lad_r50_paa_r101_fpn_coco_1x_20220708_124246.log.json) |
|  R-50   |  R-101  |        1x         |   43.2   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/lad/lad_r101_paa_r50_fpn_coco_1x.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/lad/lad_r101_paa_r50_fpn_coco_1x/lad_r101_paa_r50_fpn_coco_1x_20220708_124357-9407ac54.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/lad/lad_r101_paa_r50_fpn_coco_1x/lad_r101_paa_r50_fpn_coco_1x_20220708_124357.log.json) |

## Note

- Meaning of Config name: lad_r50(student model)\_paa(based on paa)\_r101(teacher model)\_fpn(neck)\_coco(dataset)\_1x(12 epoch).py
- Results may fluctuate by about 0.2 mAP.

## Citation

```latex
@inproceedings{nguyen2021improving,
  title={Improving Object Detection by Label Assignment Distillation},
  author={Chuong H. Nguyen and Thuy C. Nguyen and Tuan N. Tang and Nam L. H. Phan},
  booktitle = {WACV},
  year={2022}
}
```
