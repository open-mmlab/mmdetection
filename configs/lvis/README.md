# LVIS dataset

## Introduction
```
@inproceedings{gupta2019lvis,
  title={{LVIS}: A Dataset for Large Vocabulary Instance Segmentation},
  author={Gupta, Agrim and Dollar, Piotr and Girshick, Ross},
  booktitle={Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

## Common Setting
* Please follow [install guide](../../docs/install.md#install-mmdetection) to install open-mmlab forked cocoapi first.
* Run following scripts to install our forked lvis-api.
    ```
    # mmlvis is fully compatible with official lvis
    pip install mmlvis
    ```
    or
    ```
    pip install -r requirements/optional.txt
    ```
* All experiments use oversample strategy [here](../../docs/tutorials/new_dataset.md#class-balanced-dataset) with oversample threshold `1e-3`.
* The size of LVIS v0.5 is half of COCO, so schedule `2x` in LVIS is roughly the same iterations as `1x` in COCO.

## Results and models of LVIS v0.5

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: |
|    R-50-FPN     | pytorch |   2x    | -        | -              | 26.1   | 25.9    | [model](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis-dbd06831.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis_20200531_160435.log.json)  |
|    R-101-FPN    | pytorch |   2x    | -        | -              | 27.1   | 27.0    | [model](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_2x_lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_2x_lvis-54582ee2.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_2x_lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_2x_lvis_20200601_134748.log.json)  |
| X-101-32x4d-FPN | pytorch |   2x    | -        | -              | 26.7   | 26.9    | [model](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_2x_lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_2x_lvis-3cf55ea2.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_2x_lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_2x_lvis_20200531_221749.log.json)  |
| X-101-64x4d-FPN | pytorch |   2x    | -        |   -            | 26.4   | 26.0    | [model](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_2x_lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_2x_lvis-1c99a5ad.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_2x_lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_2x_lvis_20200601_194651.log.json)  |

## Results and models of LVIS v1

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: |
|    R-50-FPN     | pytorch |   1x    | 9.1      | -              | 22.5   | 21.7    | [model](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1-aa78ac3d.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1-20200829_061305.log.json)  |
|    R-101-FPN    | pytorch |   1x    | 10.8     | -              | 24.6   | 23.6    | [model](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_r101_fpn_sample1e-3_mstrain_1x_lvis_v1-ec55ce32.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_r101_fpn_sample1e-3_mstrain_1x_lvis_v1-20200829_070959.log.json)  |
| X-101-32x4d-FPN | pytorch |   1x    | 11.8     | -              | 26.7   | 25.5    | [model](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_1x_lvis_v1-ebbc5c81.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_1x_lvis_v1-20200829_071317.log.json)  |
| X-101-64x4d-FPN | pytorch |   1x    | 14.6     | -              | 27.2   | 25.8    | [model](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_1x_lvis_v1-43d9edfe.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_1x_lvis_v1-20200830_060206.log.json)  |
