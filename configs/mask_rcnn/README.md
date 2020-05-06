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
|    R-50-FPN     |  caffe  |   1x    |          |                |        |         |   |
|    R-50-FPN     | pytorch |   1x    | 4.4      | 12.0           | 38.2   | 34.7    |  [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205_050542.log.json) |
|    R-50-FPN     | pytorch |   2x    |          |                |        |         |   |
|    R-101-FPN    |  caffe  |   1x    |          |                |        |         |   |
|    R-101-FPN    | pytorch |   1x    | 6.4      | 10.3           | 40.0   | 36.1    |  [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_1x_coco/mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_1x_coco/mask_rcnn_r101_fpn_1x_coco_20200204_144809.log.json) |
|    R-101-FPN    | pytorch |   2x    |          |                |        |         |   |
| X-101-32x4d-FPN | pytorch |   1x    | 7.6      | 9.4            | 41.9   | 37.5    |  [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x4d_fpn_1x_coco/mask_rcnn_x101_32x4d_fpn_1x_coco_20200205-478d0b67.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x4d_fpn_1x_coco/mask_rcnn_x101_32x4d_fpn_1x_coco_20200205_034906.log.json) |
| X-101-32x4d-FPN | pytorch |   2x    |          |                |        |         |   |
| X-101-64x4d-FPN | pytorch |   1x    | 10.7     | 6.9            | 42.8   | 38.4    |  [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_1x_coco/mask_rcnn_x101_64x4d_fpn_1x_coco_20200201-9352eb0d.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_1x_coco/mask_rcnn_x101_64x4d_fpn_1x_coco_20200201_124310.log.json) |
| X-101-64x4d-FPN | pytorch |   2x    |          |                |        |         |   |
