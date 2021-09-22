# SOLO: Segmenting Objects by Locations

## Introduction

```
@inproceedings{wang2020solo,
  title     =  {{SOLO}: Segmenting Objects by Locations},
  author    =  {Wang, Xinlong and Kong, Tao and Shen, Chunhua and Jiang, Yuning and Li, Lei},
  booktitle =  {Proc. Eur. Conf. Computer Vision (ECCV)},
  year      =  {2020}
}
```

## Results and Models

### SOLO

| Backbone  | Style   | MS train | Lr schd | Mem (GB) | Inf time (fps) | mask AP | Download |
|:---------:|:-------:|:--------:|:-------:|:--------:|:--------------:|:------:|:--------:|
| R-50      | pytorch   | N        | 1x      |  8.0   |   95.1     |  33.1  |  [model](https://download.openmmlab.com/mmdetection/v2.0/solo/solo_r50_fpn_1x_coco/solo_r50_fpn_1x_coco_20210821_035055-2290a6b8.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/solo/solo_r50_fpn_1x_coco/solo_r50_fpn_1x_coco_20210821_035055.log.json) |
| R-50      | pytorch   | Y        | 3x      |  7.4   |   94.2  |  35.9  |  [model](https://download.openmmlab.com/mmdetection/v2.0/solo/solo_r50_fpn_3x_coco/solo_r50_fpn_3x_coco_20210901_012353-11d224d7.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/solo/solo_r50_fpn_3x_coco/solo_r50_fpn_3x_coco_20210901_012353.log.json) |

### Decoupled SOLO

| Backbone  | Style   | MS train | Lr schd | Mem (GB) | Inf time (fps) | mask AP | Download |
|:---------:|:-------:|:--------:|:-------:|:--------:|:--------------:|:------:|:--------:|
| R-50      | pytorch   | N        | 1x      |  7.8  |    116.4   |  33.9  |  [model](https://download.openmmlab.com/mmdetection/v2.0/solo/decoupled_solo_r50_fpn_1x_coco/decoupled_solo_r50_fpn_1x_coco_20210820_233348-6337c589.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/solo/decoupled_solo_r50_fpn_1x_coco/decoupled_solo_r50_fpn_1x_coco_20210820_233348.log.json) |
| R-50      | pytorch   | Y        | 3x      |  7.9 |   117.2   |  36.7  |  [model](https://download.openmmlab.com/mmdetection/v2.0/solo/decoupled_solo_r50_fpn_3x_coco/decoupled_solo_r50_fpn_3x_coco_20210821_042504-7b3301ec.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/solo/decoupled_solo_r50_fpn_3x_coco/decoupled_solo_r50_fpn_3x_coco_20210821_042504.log.json) |

### Decoupled Light SOLO

| Backbone  | Style   | MS train | Lr schd | Mem (GB) | Inf time (fps) | mask AP | Download |
|:---------:|:-------:|:--------:|:-------:|:--------:|:--------------:|:------:|:--------:|
| R-50      | pytorch   | Y        | 3x      | 2.2    |    23.9    | 32.9  |  [model](https://download.openmmlab.com/mmdetection/v2.0/solo/decoupled_solo_light_r50_fpn_3x_coco/decoupled_solo_light_r50_fpn_3x_coco_20210906_142703-e70e226f.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/solo/decoupled_solo_light_r50_fpn_3x_coco/decoupled_solo_light_r50_fpn_3x_coco_20210906_142703.log.json) |
