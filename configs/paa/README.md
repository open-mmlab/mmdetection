# Probabilistic Anchor Assignment with IoU Prediction for Object Detection

[ALGORITHM]

## Results and Models

We provide config files to reproduce the object detection results in the
ECCV 2020 paper for Probabilistic Anchor Assignment with IoU
Prediction for Object Detection.

| Backbone    | Lr schd | Mem (GB) | Score voting | box AP | Config | Download |
|:-----------:|:-------:|:--------:|:------------:|:------:|:------:|:--------:|
| R-50-FPN    | 12e     | 3.7     | True          | 40.4   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/paa/paa_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_1x_coco/paa_r50_fpn_1x_coco_20200821-936edec3.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_1x_coco/paa_r50_fpn_1x_coco_20200821-936edec3.log.json) |
| R-50-FPN    | 12e     | 3.7     | False         | 40.2   | - |
| R-50-FPN    | 18e     | 3.7     | True          | 41.4   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/paa/paa_r50_fpn_1.5x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_1.5x_coco/paa_r50_fpn_1.5x_coco_20200823-805d6078.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_1.5x_coco/paa_r50_fpn_1.5x_coco_20200823-805d6078.log.json) |
| R-50-FPN    | 18e     | 3.7     | False         | 41.2   | - |
| R-50-FPN    | 24e     | 3.7     | True          | 41.6   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/paa/paa_r50_fpn_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_2x_coco/paa_r50_fpn_2x_coco_20200821-c98bfc4e.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_2x_coco/paa_r50_fpn_2x_coco_20200821-c98bfc4e.log.json) |
| R-50-FPN    | 36e     | 3.7     | True          | 43.3   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/paa/paa_r50_fpn_mstrain_3x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_mstrain_3x_coco/paa_r50_fpn_mstrain_3x_coco_20210121_145722-06a6880b.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_mstrain_3x_coco/paa_r50_fpn_mstrain_3x_coco_20210121_145722.log.json) |
| R-101-FPN   | 12e     | 6.2     | True          | 42.6   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/paa/paa_r101_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r101_fpn_1x_coco/paa_r101_fpn_1x_coco_20200821-0a1825a4.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r101_fpn_1x_coco/paa_r101_fpn_1x_coco_20200821-0a1825a4.log.json) |
| R-101-FPN   | 12e     | 6.2     | False         | 42.4   | - |
| R-101-FPN   | 24e     | 6.2     | True          | 43.5   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/paa/paa_r101_fpn_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r101_fpn_2x_coco/paa_r101_fpn_2x_coco_20200821-6829f96b.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r101_fpn_2x_coco/paa_r101_fpn_2x_coco_20200821-6829f96b.log.json) |
| R-101-FPN   | 36e     | 6.2     | True          | 45.1   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/paa/paa_r101_fpn_mstrain_3x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r101_fpn_mstrain_3x_coco/paa_r101_fpn_mstrain_3x_coco_20210122_084202-83250d22.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r101_fpn_mstrain_3x_coco/paa_r101_fpn_mstrain_3x_coco_20210122_084202.log.json) |

**Note**:

1. We find that the performance is unstable with 1x setting and may fluctuate by about 0.2 mAP. We report the best results.
