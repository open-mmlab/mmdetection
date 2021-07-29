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

| Backbone    | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:-----------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| PVT-Tiny    | 12e     |          |                |        | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pvt/retinanet_pvt_t_fpn_1x_coco.py) | [model]() &#124; [log]() |
| PVT-Small   | 12e     |          |                |        | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pvt/retinanet_pvt_s_fpn_1x_coco.py) | [model]() &#124; [log]() |
| PVT-Medium  | 12e     |          |                |        | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pvt/retinanet_pvt_m_fpn_1x_coco.py) | [model]() &#124; [log]() |
| PVT-Large   | 12e     |          |                |        | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pvt/retinanet_pvt_l_fpn_1x_coco.py) | [model]() &#124; [log]() |

### RetinaNet (PVTv2)

| Backbone    | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:-----------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| PVTv2-B0    | 12e     |          |                |        | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pvt/retinanet_pvt_v2_b0_fpn_1x_coco.py) | [model]() &#124; [log]() |
| PVTv2-B1    | 12e     |          |                |        | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pvt/retinanet_pvt_v2_b1_fpn_1x_coco.py) | [model]() &#124; [log]() |
| PVTv2-B2    | 12e     |          |                |        | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pvt/retinanet_pvt_v2_b2_fpn_1x_coco.py) | [model]() &#124; [log]() |
| PVTv2-B3    | 12e     |          |                |        | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pvt/retinanet_pvt_v2_b3_fpn_1x_coco.py) | [model]() &#124; [log]() |
| PVTv2-B4    | 12e     |          |                |        | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pvt/retinanet_pvt_v2_b4_fpn_1x_coco.py) | [model]() &#124; [log]() |
| PVTv2-B5    | 12e     |          |                |        | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pvt/retinanet_pvt_v2_b5_fpn_1x_coco.py) | [model]() &#124; [log]() |
