# Prime Sample Attention in Object Detection

## Introduction

```
@inproceedings{cao2019prime,
  title={Prime sample attention in object detection},
  author={Cao, Yuhang and Chen, Kai and Loy, Chen Change and Lin, Dahua},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

## Results and models


| PISA | Network | Backbone            | Lr schd | box AP | mask AP | Download |
|:----:|:-------:|:-------------------:|:-------:|:------:|:-------:|:--------:|
| ×    | Faster R-CNN | R-50-FPN       | 1x      | 36.4   |         | - |
| √    | Faster R-CNN | R-50-FPN       | 1x      | 38.4   |         | [model](http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_faster_rcnn_r50_fpn_1x_coco/pisa_faster_rcnn_r50_fpn_1x_coco-dea93523.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_faster_rcnn_r50_fpn_1x_coco/pisa_faster_rcnn_r50_fpn_1x_coco_20200506_185619.log.json)  |
| ×    | Faster R-CNN | X101-32x4d-FPN | 1x      | 40.1   |         | - |
| √    | Faster R-CNN | X101-32x4d-FPN | 1x      | 41.9   |         | [model](http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_faster_rcnn_x101_32x4d_fpn_1x_coco/pisa_faster_rcnn_x101_32x4d_fpn_1x_coco-e4accec4.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_faster_rcnn_x101_32x4d_fpn_1x_coco/pisa_faster_rcnn_x101_32x4d_fpn_1x_coco_20200505_181503.log.json) |
| ×    | Mask   R-CNN | R-50-FPN       | 1x      | 37.3   | 34.2    | - |
| √    | Mask   R-CNN | R-50-FPN       | 1x      | 39.1   | 35.2    | [model](http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_mask_rcnn_r50_fpn_1x_coco/pisa_mask_rcnn_r50_fpn_1x_coco-dfcedba6.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_mask_rcnn_r50_fpn_1x_coco/pisa_mask_rcnn_r50_fpn_1x_coco_20200508_150500.log.json) |
| ×    | Mask   R-CNN | X101-32x4d-FPN | 1x      | 41.1   | 37.1    | - |
| √    | Mask   R-CNN | X101-32x4d-FPN | 1x      |        |         |   |
| ×    | RetinaNet    | R-50-FPN       | 1x      | 35.6   |         | - |
| √    | RetinaNet    | R-50-FPN       | 1x      | 36.9   |         | [model](http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_retinanet_r50_fpn_1x_coco/pisa_retinanet_r50_fpn_1x_coco-76409952.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_retinanet_r50_fpn_1x_coco/pisa_retinanet_r50_fpn_1x_coco_20200504_014311.log.json) |
| ×    | RetinaNet    | X101-32x4d-FPN | 1x      | 39.0   |         | - |
| √    | RetinaNet    | X101-32x4d-FPN | 1x      | 40.7   |         | [model](http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_retinanet_x101_32x4d_fpn_1x_coco/pisa_retinanet_x101_32x4d_fpn_1x_coco-a0c13c73.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_retinanet_x101_32x4d_fpn_1x_coco/pisa_retinanet_x101_32x4d_fpn_1x_coco_20200505_001404.log.json) |
| ×    | SSD300       | VGG16          | 1x      | 25.6   |         | - |
| √    | SSD300       | VGG16          | 1x      | 27.6   |         | [model](http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_ssd300_coco/pisa_ssd300_coco-710e3ac9.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_ssd300_coco/pisa_ssd300_coco_20200504_144325.log.json) |
| ×    | SSD300       | VGG16          | 1x      | 29.3   |         | - |
| √    | SSD300       | VGG16          | 1x      | 31.8   |         | [model](http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_ssd512_coco/pisa_ssd512_coco-247addee.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_ssd512_coco/pisa_ssd512_coco_20200508_131030.log.json)  |

**Notes:**
- In the original paper, all models are trained and tested on mmdet v1.x, thus results may not be exactly the same with this release on v2.0.
- It is noted PISA only modifies the training pipeline so the inference time remains the same with the baseline.
