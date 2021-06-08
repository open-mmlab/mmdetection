# Region Proposal by Guided Anchoring

## Introduction

<!-- [ALGORITHM] -->

We provide config files to reproduce the results in the CVPR 2019 paper for [Region Proposal by Guided Anchoring](https://arxiv.org/abs/1901.03278).

```latex
@inproceedings{wang2019region,
    title={Region Proposal by Guided Anchoring},
    author={Jiaqi Wang and Kai Chen and Shuo Yang and Chen Change Loy and Dahua Lin},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
    year={2019}
}
```

## Results and Models

The results on COCO 2017 val is shown in the below table. (results on test-dev are usually slightly higher than val).

| Method |    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | AR 1000 | Config | Download |
| :----: | :-------------: | :-----: | :-----: | :------: | :------------: | :-----: | :------: | :--------: |
| GA-RPN |    R-50-FPN     |  caffe  |   1x    |   5.3    |      15.8      |  68.4   |   [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/guided_anchoring/ga_rpn_r50_caffe_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_rpn_r50_caffe_fpn_1x_coco/ga_rpn_r50_caffe_fpn_1x_coco_20200531-899008a6.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_rpn_r50_caffe_fpn_1x_coco/ga_rpn_r50_caffe_fpn_1x_coco_20200531_011819.log.json)   |
| GA-RPN |    R-101-FPN    |  caffe  |   1x    |   7.3    |      13.0      |  69.5   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/guided_anchoring/ga_rpn_r101_caffe_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_rpn_r101_caffe_fpn_1x_coco/ga_rpn_r101_caffe_fpn_1x_coco_20200531-ca9ba8fb.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_rpn_r101_caffe_fpn_1x_coco/ga_rpn_r101_caffe_fpn_1x_coco_20200531_011812.log.json) |
| GA-RPN | X-101-32x4d-FPN | pytorch |   1x    |   8.5    |      10.0      |  70.6   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/guided_anchoring/ga_rpn_x101_32x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_rpn_x101_32x4d_fpn_1x_coco/ga_rpn_x101_32x4d_fpn_1x_coco_20200220-c28d1b18.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_rpn_x101_32x4d_fpn_1x_coco/ga_rpn_x101_32x4d_fpn_1x_coco_20200220_221326.log.json) |
| GA-RPN | X-101-64x4d-FPN | pytorch |   1x    |   7.1    |      7.5       |  71.2   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/guided_anchoring/ga_rpn_x101_64x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_rpn_x101_64x4d_fpn_1x_coco/ga_rpn_x101_64x4d_fpn_1x_coco_20200225-3c6e1aa2.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_rpn_x101_64x4d_fpn_1x_coco/ga_rpn_x101_64x4d_fpn_1x_coco_20200225_152704.log.json) |

|     Method     |    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :------------: | :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
| GA-Faster RCNN |    R-50-FPN     |  caffe  |   1x    |   5.5    |                |  39.6  |          [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/guided_anchoring/ga_faster_r50_caffe_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_faster_r50_caffe_fpn_1x_coco/ga_faster_r50_caffe_fpn_1x_coco_20200702_000718-a11ccfe6.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_faster_r50_caffe_fpn_1x_coco/ga_faster_r50_caffe_fpn_1x_coco_20200702_000718.log.json)           |
| GA-Faster RCNN |    R-101-FPN    |  caffe  |   1x    |   7.5    |                |  41.5  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/guided_anchoring/ga_faster_r101_caffe_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_faster_r101_caffe_fpn_1x_coco/ga_faster_r101_caffe_fpn_1x_coco_bbox_mAP-0.415_20200505_115528-fb82e499.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_faster_r101_caffe_fpn_1x_coco/ga_faster_r101_caffe_fpn_1x_coco_20200505_115528.log.json) |
| GA-Faster RCNN | X-101-32x4d-FPN | pytorch |   1x    |   8.7    |      9.7       |  43.0  |            [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/guided_anchoring/ga_faster_x101_32x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_faster_x101_32x4d_fpn_1x_coco/ga_faster_x101_32x4d_fpn_1x_coco_20200215-1ded9da3.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_faster_x101_32x4d_fpn_1x_coco/ga_faster_x101_32x4d_fpn_1x_coco_20200215_184547.log.json)            |
| GA-Faster RCNN | X-101-64x4d-FPN | pytorch |   1x    |   11.8   |      7.3       |  43.9  |            [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/guided_anchoring/ga_faster_x101_64x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_faster_x101_64x4d_fpn_1x_coco/ga_faster_x101_64x4d_fpn_1x_coco_20200215-0fa7bde7.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_faster_x101_64x4d_fpn_1x_coco/ga_faster_x101_64x4d_fpn_1x_coco_20200215_104455.log.json)            |
|  GA-RetinaNet  |    R-50-FPN     |  caffe  |   1x    |   3.5    |      16.8      |  36.9  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/guided_anchoring/ga_retinanet_r50_caffe_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_retinanet_r50_caffe_fpn_1x_coco/ga_retinanet_r50_caffe_fpn_1x_coco_20201020-39581c6f.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_retinanet_r50_caffe_fpn_1x_coco/ga_retinanet_r50_caffe_fpn_1x_coco_20201020_225450.log.json)       |
|  GA-RetinaNet  |    R-101-FPN    |  caffe  |   1x    |   5.5    |      12.9      |  39.0  |      [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/guided_anchoring/ga_retinanet_r101_caffe_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_retinanet_r101_caffe_fpn_1x_coco/ga_retinanet_r101_caffe_fpn_1x_coco_20200531-6266453c.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_retinanet_r101_caffe_fpn_1x_coco/ga_retinanet_r101_caffe_fpn_1x_coco_20200531_012847.log.json)      |
|  GA-RetinaNet  | X-101-32x4d-FPN | pytorch |   1x    |   6.9    |      10.6      |  40.5  |      [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/guided_anchoring/ga_retinanet_x101_32x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_retinanet_x101_32x4d_fpn_1x_coco/ga_retinanet_x101_32x4d_fpn_1x_coco_20200219-40c56caa.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_retinanet_x101_32x4d_fpn_1x_coco/ga_retinanet_x101_32x4d_fpn_1x_coco_20200219_223025.log.json)      |
|  GA-RetinaNet  | X-101-64x4d-FPN | pytorch |   1x    |   9.9    |      7.7       |  41.3  |      [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/guided_anchoring/ga_retinanet_x101_64x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_retinanet_x101_64x4d_fpn_1x_coco/ga_retinanet_x101_64x4d_fpn_1x_coco_20200226-ef9f7f1f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_retinanet_x101_64x4d_fpn_1x_coco/ga_retinanet_x101_64x4d_fpn_1x_coco_20200226_221123.log.json)      |

- In the Guided Anchoring paper, `score_thr` is set to 0.001 in Fast/Faster RCNN and 0.05 in RetinaNet for both baselines and Guided Anchoring.

- Performance on COCO test-dev benchmark are shown as follows.

|     Method     | Backbone  | Style | Lr schd | Aug Train | Score thr |  AP   | AP_50 | AP_75 | AP_small | AP_medium | AP_large | Download |
| :------------: | :-------: | :---: | :-----: | :-------: | :-------: | :---: | :---: | :---: | :------: | :-------: | :------: | :------: |
| GA-Faster RCNN | R-101-FPN | caffe |   1x    |     F     |   0.05    |       |       |       |          |           |          |          |
| GA-Faster RCNN | R-101-FPN | caffe |   1x    |     F     |   0.001   |       |       |       |          |           |          |          |
|  GA-RetinaNet  | R-101-FPN | caffe |   1x    |     F     |   0.05    |       |       |       |          |           |          |          |
|  GA-RetinaNet  | R-101-FPN | caffe |   2x    |     T     |   0.05    |       |       |       |          |           |          |          |
