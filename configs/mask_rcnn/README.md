# Mask R-CNN

## Introduction
```
@article{He_2017,
   title={Mask R-CNN},
   journal={2017 IEEE International Conference on Computer Vision (ICCV)},
   publisher={IEEE},
   author={He, Kaiming and Gkioxari, Georgia and Dollar, Piotr and Girshick, Ross},
   year={2017},
   month={Oct}
}
```

## Results and models

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: |
|    R-50-FPN     |  caffe  |   1x    | 4.3      |                | 38.0   | 34.4    |  [model](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_1x_coco/mask_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.38__segm_mAP-0.344_20200504_231812-0ebd1859.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_1x_coco/mask_rcnn_r50_caffe_fpn_1x_coco_20200504_231812.log.json) |
|    R-50-FPN     | pytorch |   1x    | 4.4      | 16.1           | 38.2   | 34.7    |  [model](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205_050542.log.json) |
|    R-50-FPN     | pytorch |   2x    | -        | -              | 39.2   | 35.4    | [model](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_2x_coco/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_2x_coco/mask_rcnn_r50_fpn_2x_coco_20200505_003907.log.json)  |
|    R-101-FPN    |  caffe  |   1x    |          |                | 40.4   | 36.4    |  [model](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_caffe_fpn_1x_coco/mask_rcnn_r101_caffe_fpn_1x_coco_20200601_095758-805e06c1.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_caffe_fpn_1x_coco/mask_rcnn_r101_caffe_fpn_1x_coco_20200601_095758.log.json)|
|    R-101-FPN    | pytorch |   1x    | 6.4      | 13.5           | 40.0   | 36.1    |  [model](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_1x_coco/mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_1x_coco/mask_rcnn_r101_fpn_1x_coco_20200204_144809.log.json) |
|    R-101-FPN    | pytorch |   2x    | -        | -              | 40.8   | 36.6    | [model](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_2x_coco/mask_rcnn_r101_fpn_2x_coco_bbox_mAP-0.408__segm_mAP-0.366_20200505_071027-14b391c7.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_2x_coco/mask_rcnn_r101_fpn_2x_coco_20200505_071027.log.json)  |
| X-101-32x4d-FPN | pytorch |   1x    | 7.6      | 11.3           | 41.9   | 37.5    |  [model](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x4d_fpn_1x_coco/mask_rcnn_x101_32x4d_fpn_1x_coco_20200205-478d0b67.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x4d_fpn_1x_coco/mask_rcnn_x101_32x4d_fpn_1x_coco_20200205_034906.log.json) |
| X-101-32x4d-FPN | pytorch |   2x    | -        | -              | 42.2   | 37.8    | [model](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x4d_fpn_2x_coco/mask_rcnn_x101_32x4d_fpn_2x_coco_bbox_mAP-0.422__segm_mAP-0.378_20200506_004702-faef898c.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x4d_fpn_2x_coco/mask_rcnn_x101_32x4d_fpn_2x_coco_20200506_004702.log.json)  |
| X-101-64x4d-FPN | pytorch |   1x    | 10.7     | 8.0            | 42.8   | 38.4    |  [model](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_1x_coco/mask_rcnn_x101_64x4d_fpn_1x_coco_20200201-9352eb0d.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_1x_coco/mask_rcnn_x101_64x4d_fpn_1x_coco_20200201_124310.log.json) |
| X-101-64x4d-FPN | pytorch |   2x    |  -       |   -            |  42.7  |  38.1   | [model](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_2x_coco/mask_rcnn_x101_64x4d_fpn_2x_coco_20200509_224208-39d6f70c.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_2x_coco/mask_rcnn_x101_64x4d_fpn_2x_coco_20200509_224208.log.json)|
| X-101-32x8d-FPN | pytorch |   1x    |  -       |   -            |  42.8  |  38.3   | |


## Pre-trained Models
We also train some models with longer schedules and multi-scale training. The users could finetune them for downstream tasks.

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: |
|    [R-50-FPN](./mask_rcnn_r50_caffe_fpn_mstrain-poly_2x_coco.py)     |  caffe  |   2x    | 4.3      |                | 40.3   | 36.5    |  [model](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_2x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_2x_coco_bbox_mAP-0.403__segm_mAP-0.365_20200504_231822-a75c98ce.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_2x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_2x_coco_20200504_231822.log.json)
|    [R-50-FPN](./mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py)     |  caffe  |   3x    | 4.3      |                | 40.8   | 37.0    |  [model](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_20200504_163245.log.json)
|    [X-101-32x8d-FPN](./mask_rcnn_x101_32x8d_fpn_mstrain-poly_3x_coco.py)     |  pytorch  |   1x    | -     |       | 43.6 | 39.0 |
|    [X-101-32x8d-FPN](./mask_rcnn_x101_32x8d_fpn_mstrain-poly_3x_coco.py)     |  pytorch  |   3x    | -     |       | 44.0 | 39.3 |
