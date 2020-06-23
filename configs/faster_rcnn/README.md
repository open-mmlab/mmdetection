# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

## Introduction
```
@article{Ren_2017,
   title={Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks},
   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
   year={2017},
   month={Jun},
}
```

## Results and models

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: |
|    R-50-FPN     |  caffe  |   1x    | 3.8      |                | 37.8   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_1x_coco/faster_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.378_20200504_180032-c5925ee5.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_1x_coco/faster_rcnn_r50_caffe_fpn_1x_coco_20200504_180032.log.json) |
|    R-50-FPN     | pytorch |   1x    | 4.0      | 21.4           | 37.4   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130_204655.log.json) |
|    R-50-FPN     | pytorch |   2x    | -        | -              | 38.4   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_20200504_210434.log.json) |
|    R-101-FPN    |  caffe  |   1x    | 5.7      |                | 39.8   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_caffe_fpn_1x_coco/faster_rcnn_r101_caffe_fpn_1x_coco_bbox_mAP-0.398_20200504_180057-b269e9dd.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_caffe_fpn_1x_coco/faster_rcnn_r101_caffe_fpn_1x_coco_20200504_180057.log.json) |
|    R-101-FPN    | pytorch |   1x    | 6.0      | 15.6           | 39.4   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_1x_coco/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_1x_coco/faster_rcnn_r101_fpn_1x_coco_20200130_204655.log.json) |
|    R-101-FPN    | pytorch |   2x    | -        | -              | 39.8   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_2x_coco/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_2x_coco/faster_rcnn_r101_fpn_2x_coco_20200504_210455.log.json) |
| X-101-32x4d-FPN | pytorch |   1x    | 7.2      | 13.8           | 41.2   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_32x4d_fpn_1x_coco/faster_rcnn_x101_32x4d_fpn_1x_coco_20200203-cff10310.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_32x4d_fpn_1x_coco/faster_rcnn_x101_32x4d_fpn_1x_coco_20200203_000520.log.json) |
| X-101-32x4d-FPN | pytorch |   2x    | -        | -              | 41.2   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_32x4d_fpn_2x_coco/faster_rcnn_x101_32x4d_fpn_2x_coco_bbox_mAP-0.412_20200506_041400-64a12c0b.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_32x4d_fpn_2x_coco/faster_rcnn_x101_32x4d_fpn_2x_coco_20200506_041400.log.json) |
| X-101-64x4d-FPN | pytorch |   1x    | 10.3     | 9.4            | 42.1   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_64x4d_fpn_1x_coco/faster_rcnn_x101_64x4d_fpn_1x_coco_20200204-833ee192.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_64x4d_fpn_1x_coco/faster_rcnn_x101_64x4d_fpn_1x_coco_20200204_134340.log.json) |
| X-101-64x4d-FPN | pytorch |   2x    | -        | -              | 41.6   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_64x4d_fpn_2x_coco/faster_rcnn_x101_64x4d_fpn_2x_coco_20200512_161033-5961fa95.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_64x4d_fpn_2x_coco/faster_rcnn_x101_64x4d_fpn_2x_coco_20200512_161033.log.json)  |

## Different regression loss
We trained with R-50-FPN pytorch style backbone for 1x schedule.

|    Backbone     | Loss type | Mem (GB) | Inf time (fps) | box AP | Download |
| :-------------: | :-------: | :------: | :------------: | :----: | :------: |
|    R-50-FPN     |  L1Loss   | 4.0      | 21.4           | 37.4   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130_204655.log.json) |
|    R-50-FPN     |  IoULoss  |          |                | 37.9   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_iou_1x_coco-fdd207f3.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_iou_1x_coco_20200506_095954.log.json)  |
|    R-50-FPN     |  GIoULoss |          |                | 37.6   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_giou_1x_coco-0eada910.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_giou_1x_coco_20200505_161120.log.json)  |
|    R-50-FPN     |  BoundedIoULoss |          |                | 37.4   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_bounded_iou_1x_coco-98ad993b.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_bounded_iou_1x_coco_20200505_160738.log.json)  |

## Pre-trained Models
We also train some models with longer schedules and multi-scale training. The users could finetune them for downstream tasks.

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: |
|    [R-50-FPN](./faster_rcnn_r50_caffe_fpn_mstrain-poly_2x_coco.py)     |  caffe  |   2x    | 4.3      |                | 39.7   |[model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_2x_coco/faster_rcnn_r50_caffe_fpn_mstrain_2x_coco_bbox_mAP-0.397_20200504_231813-10b2de58.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_2x_coco/faster_rcnn_r50_caffe_fpn_mstrain_2x_coco_20200504_231813.log.json)
|    [R-50-FPN](./faster_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py)     |  caffe  |   3x    | 4.3      |                | 40.2   |  [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_bbox_mAP-0.398_20200504_163323-30042637.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20200504_163323.log.json)
