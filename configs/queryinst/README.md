# Instances as Queries

## Introduction

<!-- [ALGORITHM] -->

```
@InProceedings{Fang_2021_ICCV,
    author    = {Fang, Yuxin and Yang, Shusheng and Wang, Xinggang and Li, Yu and Fang, Chen and Shan, Ying and Feng, Bin and Liu, Wenyu},
    title     = {Instances As Queries},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {6910-6919}
}
```

## Results and Models

| Model        | Backbone  | Style   | Lr schd | Number of Proposals |Multi-Scale| RandomCrop  | box AP  | mask AP  | Config | Download |
|:------------:|:---------:|:-------:|:-------:|:-------:            |:-------: |:---------:|:------:|:------:|:------:|:--------:|
| QueryInst | R-50-FPN  | pytorch | 1x      |   100               | False     |  False     |  42.0  |  37.5  |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/queryinst/queryinst_r50_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/queryinst/queryinst_r50_fpn_1x_coco/queryinst_r50_fpn_1x_coco_20210907_084916-5a8f1998.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/queryinst/queryinst_r50_fpn_1x_coco/queryinst_r50_fpn_1x_coco_20210907_084916.log.json) |
| QueryInst | R-50-FPN  | pytorch | 3x      |   100               | True     |   False     |  44.8  |  39.8  |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/queryinst/queryinst_r50_fpn_mstrain_480-800_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/queryinst/queryinst_r50_fpn_mstrain_480-800_3x_coco/queryinst_r50_fpn_mstrain_480-800_3x_coco_20210901_103643-7837af86.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/queryinst/queryinst_r50_fpn_mstrain_480-800_3x_coco/queryinst_r50_fpn_mstrain_480-800_3x_coco_20210901_103643.log.json) |
| QueryInst | R-50-FPN  | pytorch | 3x      |   300               | True      |  True      |  47.5  |  41.7  |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/queryinst/queryinst_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/queryinst/queryinst_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco/queryinst_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20210904_101802-85cffbd8.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/queryinst/queryinst_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco/queryinst_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20210904_101802.log.json) |
| QueryInst | R-101-FPN | pytorch | 3x      |   100               | True      |  False     |  46.4  |  41.0  |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/queryinst/queryinst_r101_fpn_mstrain_480-800_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/queryinst/queryinst_r101_fpn_mstrain_480-800_3x_coco/queryinst_r101_fpn_mstrain_480-800_3x_coco_20210904_104048-91f9995b.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/queryinst/queryinst_r101_fpn_mstrain_480-800_3x_coco/queryinst_r101_fpn_mstrain_480-800_3x_coco_20210904_104048.log.json) |
| QueryInst | R-101-FPN | pytorch | 3x      |   300               | True      |  True      |  49.0  |  42.9  |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/queryinst/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/queryinst/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20210904_153621-76cce59f.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/queryinst/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20210904_153621.log.json) |
