# ResNeSt: Split-Attention Networks

## Introduction

[BACKBONE]

```latex
@article{zhang2020resnest,
title={ResNeSt: Split-Attention Networks},
author={Zhang, Hang and Wu, Chongruo and Zhang, Zhongyue and Zhu, Yi and Zhang, Zhi and Lin, Haibin and Sun, Yue and He, Tong and Muller, Jonas and Manmatha, R. and Li, Mu and Smola, Alexander},
journal={arXiv preprint arXiv:2004.08955},
year={2020}
}
```

## Results and Models

### Faster R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|S-50-FPN       | pytorch |   1x   |   4.8  |   -           | 42.0 |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/resnest/faster_rcnn_s50_fpn_syncbn-backbone+head_mstrain-range_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/resnest/faster_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco/faster_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco_20200926_125502-20289c16.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/resnest/faster_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco/faster_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco-20200926_125502.log.json) |
|S-101-FPN       | pytorch |   1x   |   7.1  |   -           | 44.5 |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/resnest/faster_rcnn_s101_fpn_syncbn-backbone+head_mstrain-range_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/resnest/faster_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco/faster_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco_20201006_021058-421517f1.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/resnest/faster_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco/faster_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco-20201006_021058.log.json) |

### Mask R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: | :--------: |
|S-50-FPN       | pytorch |    1x   |   5.5  |      -         | 42.6 | 38.1 |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/resnest/mask_rcnn_s50_fpn_syncbn-backbone+head_mstrain_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/resnest/mask_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco/mask_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco_20200926_125503-8a2c3d47.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/resnest/mask_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco/mask_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco-20200926_125503.log.json) |
|S-101-FPN       | pytorch |    1x   |   7.8  |      -         | 45.2 | 40.2 |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/resnest/mask_rcnn_s101_fpn_syncbn-backbone+head_mstrain_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/resnest/mask_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco/mask_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco_20201005_215831-af60cdf9.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/resnest/mask_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco/mask_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco-20201005_215831.log.json) |

### Cascade R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|S-50-FPN       | pytorch |   1x   |   -    |   -           |  44.5  |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/resnest/cascade_rcnn_s50_fpn_syncbn-backbone+head_mstrain-range_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/resnest/cascade_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco/cascade_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco_20201122_213640-763cc7b5.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/resnest/cascade_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco/cascade_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco-20201005_113242.log.json) |
|S-101-FPN       | pytorch |   1x   |   8.4  |   -           |  46.8  |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/resnest/cascade_rcnn_s101_fpn_syncbn-backbone+head_mstrain-range_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/resnest/cascade_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco/cascade_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco_20201005_113242-b9459f8f.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/resnest/cascade_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco/cascade_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco-20201122_213640.log.json) |

### Cascade Mask R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: | :--------: |
|S-50-FPN       | pytorch |    1x   |   -    |      -         | 45.4 | 39.5 |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/resnest/cascade_mask_rcnn_s50_fpn_syncbn-backbone+head_mstrain_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/resnest/cascade_mask_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco/cascade_mask_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco_20201122_104428-99eca4c7.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/resnest/cascade_mask_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco/cascade_mask_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco-20201122_104428.log.json) |
|S-101-FPN       | pytorch |    1x   |  10.5  |      -         | 47.7 | 41.4 |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/resnest/cascade_mask_rcnn_s101_fpn_syncbn-backbone+head_mstrain_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/resnest/cascade_mask_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco/cascade_mask_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco_20201005_113243-42607475.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/resnest/cascade_mask_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco/cascade_mask_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco-20201005_113243.log.json) |
