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
* All experiments use oversample strategy [here](../../docs/tutorials/new_dataset.md#class-balanced-dataset) with oversample threshold `1e-3`.
* The size of LVIS v0.5 is half of COCO, so schedule `2x` in LVIS is roughly the same iterations as `1x` in COCO.

## Results and models

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: |
|    R-50-FPN     | pytorch |   2x    | -        | -              |        |         | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_2x_coco/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_2x_coco/mask_rcnn_r50_fpn_2x_coco_20200505_003907.log.json)  |
|    R-101-FPN    | pytorch |   2x    | -        | -              |        |         | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_2x_coco/mask_rcnn_r101_fpn_2x_coco_bbox_mAP-0.408__segm_mAP-0.366_20200505_071027-14b391c7.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_2x_coco/mask_rcnn_r101_fpn_2x_coco_20200505_071027.log.json)  |
| X-101-32x4d-FPN | pytorch |   2x    | -        | -              |        |         | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x4d_fpn_2x_coco/mask_rcnn_x101_32x4d_fpn_2x_coco_bbox_mAP-0.422__segm_mAP-0.378_20200506_004702-faef898c.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x4d_fpn_2x_coco/mask_rcnn_x101_32x4d_fpn_2x_coco_20200506_004702.log.json)  |
| X-101-64x4d-FPN | pytorch |   2x    | -        |   -            |        |         |  |
