# Cascade R-CNN: High Quality Object Detection and Instance Segmentation

## Introduction
```
@article{Cai_2019,
   title={Cascade R-CNN: High Quality Object Detection and Instance Segmentation},
   ISSN={1939-3539},
   url={http://dx.doi.org/10.1109/tpami.2019.2956516},
   DOI={10.1109/tpami.2019.2956516},
   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Cai, Zhaowei and Vasconcelos, Nuno},
   year={2019},
   pages={1–1}
}
```

## Results and models

### Cascade R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: |:--------:|
|    R-50-FPN     |  caffe  |   1x    |          |                |        | |
|    R-50-FPN     | pytorch |   1x    |   4.4    |      14.3      |  40.3  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316_214748.log.json) |
|    R-50-FPN     | pytorch |   20e   |          |                |        | |
|    R-101-FPN    |  caffe  |   1x    |          |                |        | |
|    R-101-FPN    | pytorch |   1x    |   6.4    |      11.6      |  42.0  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco/cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco/cascade_rcnn_r101_fpn_1x_coco_20200317_101744.log.json) |
|    R-101-FPN    | pytorch |   20e   |          |                |        | |
| X-101-32x4d-FPN | pytorch |   1x    |   7.6    |      10.2      |  43.7  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_1x_coco/cascade_rcnn_x101_32x4d_fpn_1x_coco_20200316-95c2deb6.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_1x_coco/cascade_rcnn_x101_32x4d_fpn_1x_coco_20200316_055608.log.json) |
| X-101-32x4d-FPN | pytorch |   20e   |          |                |        | |
| X-101-64x4d-FPN | pytorch |   1x    |          |                |        | |
| X-101-64x4d-FPN | pytorch |   20e   |          |                |        | |


### Cascade Mask R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :----------------: |
|     R-50-C4     |  caffe  |   1x    |          |                |        |         | |
|    R-50-FPN     |  caffe  |   1x    |          |                |        |         | |
|    R-50-FPN     | pytorch |   1x    |  6.0     |  8.5           | 41.2   | 35.9    | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203_170449.log.json) |
|    R-50-FPN     | pytorch |   20e   |          |                |        |         | |
|    R-101-FPN    |  caffe  |   1x    |          |                |        |         | |
|    R-101-FPN    | pytorch |   1x    |  7.9     |  7.2           | 42.9   | 37.3    | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r101_fpn_1x_coco/cascade_mask_rcnn_r101_fpn_1x_coco_20200203-befdf6ee.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r101_fpn_1x_coco/cascade_mask_rcnn_r101_fpn_1x_coco_20200203_092521.log.json) |
|    R-101-FPN    | pytorch |   20e   |          |                |        |         | |
| X-101-32x4d-FPN | pytorch |   1x    |  9.2     |  6.8           | 44.3   | 38.3    | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_32x4d_fpn_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_1x_coco_20200201-0f411b1f.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_32x4d_fpn_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_1x_coco_20200201_052416.log.json) |
| X-101-32x4d-FPN | pytorch |   20e   |          |                |        |         | |
| X-101-64x4d-FPN | pytorch |   1x    |  12.2    |  5.7           | 45.3   | 39.2    | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco_20200203-9a2db89d.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco_20200203_044059.log.json) |
| X-101-64x4d-FPN | pytorch |   20e   |          |                |        |         | |

**Notes:**

- The `20e` schedule in Cascade (Mask) R-CNN indicates decreasing the lr at 16 and 19 epochs, with a total of 20 epochs.
