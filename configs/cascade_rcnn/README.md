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
|    R-50-FPN     |  caffe  |   1x    |          |                |        | [model]() &#124; [log]() |
|    R-50-FPN     | pytorch |   1x    |   4.4    |      14.3      |  40.3  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py/cascade_rcnn_r50_fpn_1x_coco.py_20200316-3dc56deb.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py/cascade_rcnn_r50_fpn_1x_coco.py_20200316_214748.log.json) |
|    R-50-FPN     | pytorch |   20e   |          |                |        | [model]() &#124; [log]()|
|    R-101-FPN    |  caffe  |   1x    |          |                |        | [model]() &#124; [log]() |
|    R-101-FPN    | pytorch |   1x    |   6.4    |      11.6      |  42.0  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco.py/cascade_rcnn_r101_fpn_1x_coco.py_20200317-0b6a2fbf.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco.py/cascade_rcnn_r101_fpn_1x_coco.py_20200317_101744.log.json) |
|    R-101-FPN    | pytorch |   20e   |          |                |        | [model]() &#124; [log]() |
| X-101-32x4d-FPN | pytorch |   1x    |   7.6    |      10.2      |  43.7  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_1x_coco.py/cascade_rcnn_x101_32x4d_fpn_1x_coco.py_20200316-95c2deb6.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_1x_coco.py/cascade_rcnn_x101_32x4d_fpn_1x_coco.py_20200316_055608.log.json) |
| X-101-32x4d-FPN | pytorch |   20e   |          |                |        | [model]() &#124; [log]() |
| X-101-64x4d-FPN | pytorch |   1x    |          |                |        | [model]() &#124; [log]() |
| X-101-64x4d-FPN | pytorch |   20e   |          |                |        | [model]() &#124; [log]() |


### Cascade Mask R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :----------------: |
|     R-50-C4     |  caffe  |   1x    |          |                |        |         | [model]() &#124; [log]() |
|    R-50-FPN     |  caffe  |   1x    |          |                |        |         | [model]() &#124; [log]() |
|    R-50-FPN     | pytorch |   1x    |          |                |        |         | [model]() &#124; [log]() |
|    R-50-FPN     | pytorch |   20e   |          |                |        |         | [model]() &#124; [log]() |
|    R-101-FPN    |  caffe  |   1x    |          |                |        |         | [model]() &#124; [log]() |
|    R-101-FPN    | pytorch |   1x    |          |                |        |         | [model]() &#124; [log]() |
|    R-101-FPN    | pytorch |   20e   |          |                |        |         | [model]() &#124; [log]() |
| X-101-32x4d-FPN | pytorch |   1x    |          |                |        |         | [model]() &#124; [log]() |
| X-101-32x4d-FPN | pytorch |   20e   |          |                |        |         | [model]() &#124; [log]() |
| X-101-64x4d-FPN | pytorch |   1x    |          |                |        |         | [model]() &#124; [log]() |
| X-101-64x4d-FPN | pytorch |   20e   |          |                |        |         | [model]() &#124; [log]() |

**Notes:**

- The `20e` schedule in Cascade (Mask) R-CNN indicates decreasing the lr at 16 and 19 epochs, with a total of 20 epochs.
