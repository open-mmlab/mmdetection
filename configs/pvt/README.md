# Pyramid vision transformer: A versatile backbone for dense prediction without convolutions

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{wang2021pyramid,
  title={Pyramid vision transformer: A versatile backbone for dense prediction without convolutions},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Fan, Deng-Ping and Song, Kaitao and Liang, Ding and Lu, Tong and Luo, Ping and Shao, Ling},
  journal={arXiv preprint arXiv:2102.12122},
  year={2021}
}
```

```latex
@article{wang2021pvtv2,
  title={PVTv2: Improved Baselines with Pyramid Vision Transformer},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Fan, Deng-Ping and Song, Kaitao and Liang, Ding and Lu, Tong and Luo, Ping and Shao, Ling},
  journal={arXiv preprint arXiv:2106.13797},
  year={2021}
}
```
## Results and Models

### RetinaNet (PVTv1)

| Backbone    | Lr schd | Mem (GB) | box AP | Config | Download |
|:-----------:|:-------:|:--------:|:------:|:------:|:--------:|
| PVT-Tiny    | 12e     |8.5       |36.6    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pvt/retinanet_pvt_t_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/pvt/retinanet_pvt-t_fpn_1x_coco/retinanet_pvt-t_fpn_1x_coco_20210831_103110-17b566bd.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/pvt/retinanet_pvt-t_fpn_1x_coco/retinanet_pvt-t_fpn_1x_coco_20210831_103110.log.json) |
| PVT-Small   | 12e     |14.5      |40.4    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pvt/retinanet_pvt_s_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/pvt/retinanet_pvt-s_fpn_1x_coco/retinanet_pvt-s_fpn_1x_coco_20210906_142921-b6c94a5b.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/pvt/retinanet_pvt-s_fpn_1x_coco/retinanet_pvt-s_fpn_1x_coco_20210906_142921.log.json) |
| PVT-Medium  | 12e     |20.9      |41.7    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pvt/retinanet_pvt_m_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/pvt/retinanet_pvt-m_fpn_1x_coco/retinanet_pvt-m_fpn_1x_coco_20210831_103243-55effa1b.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/pvt/retinanet_pvt-m_fpn_1x_coco/retinanet_pvt-m_fpn_1x_coco_20210831_103243.log.json) |

### RetinaNet (PVTv2)

| Backbone    | Lr schd | Mem (GB) | box AP | Config | Download |
|:-----------:|:-------:|:--------:|:------:|:------:|:--------:|
| PVTv2-B0    | 12e     |7.4       |37.1    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pvt/retinanet_pvt_v2_b0_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/pvt/retinanet_pvtv2-b0_fpn_1x_coco/retinanet_pvtv2-b0_fpn_1x_coco_20210831_103157-13e9aabe.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/pvt/retinanet_pvtv2-b0_fpn_1x_coco/retinanet_pvtv2-b0_fpn_1x_coco_20210831_103157.log.json) |
| PVTv2-B1    | 12e     |9.5       |41.2    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pvt/retinanet_pvt_v2_b1_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/pvt/retinanet_pvtv2-b1_fpn_1x_coco/retinanet_pvtv2-b1_fpn_1x_coco_20210831_103318-7e169a7d.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/pvt/retinanet_pvtv2-b1_fpn_1x_coco/retinanet_pvtv2-b1_fpn_1x_coco_20210831_103318.log.json) |
| PVTv2-B2    | 12e     |16.2      |44.6    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pvt/retinanet_pvt_v2_b2_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/pvt/retinanet_pvtv2-b2_fpn_1x_coco/retinanet_pvtv2-b2_fpn_1x_coco_20210901_174843-529f0b9a.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/pvt/retinanet_pvtv2-b2_fpn_1x_coco/retinanet_pvtv2-b2_fpn_1x_coco_20210901_174843.log.json) |
| PVTv2-B3    | 12e     |23.0      |46.0    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pvt/retinanet_pvt_v2_b3_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/pvt/retinanet_pvtv2-b3_fpn_1x_coco/retinanet_pvtv2-b3_fpn_1x_coco_20210903_151512-8357deff.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/pvt/retinanet_pvtv2-b3_fpn_1x_coco/retinanet_pvtv2-b3_fpn_1x_coco_20210903_151512.log.json) |
| PVTv2-B4    | 12e     |17.0      |46.3    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pvt/retinanet_pvt_v2_b4_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/pvt/retinanet_pvtv2-b4_fpn_1x_coco/retinanet_pvtv2-b4_fpn_1x_coco_20210901_170151-83795c86.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/pvt/retinanet_pvtv2-b4_fpn_1x_coco/retinanet_pvtv2-b4_fpn_1x_coco_20210901_170151.log.json) |
| PVTv2-B5    | 12e     |18.7      |46.1    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pvt/retinanet_pvt_v2_b5_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/pvt/retinanet_pvtv2-b5_fpn_1x_coco/retinanet_pvtv2-b5_fpn_1x_coco_20210902_201800-3420eb57.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/pvt/retinanet_pvtv2-b5_fpn_1x_coco/retinanet_pvtv2-b5_fpn_1x_coco_20210902_201800.log.json) |
