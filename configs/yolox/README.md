# YOLOX: Exceeding YOLO Series in 2021

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{yolox2021,
  title={{YOLOX}: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```

## Results and Models

| Backbone  | size   | Mem (GB) |   box AP | Config | Download |
|:---------:|:-------:|:-------:|:-------:|:--------:|:------:|
| YOLOX-Tiny | 416 |   3.6      |   31.6  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox/yolox_tiny_8x8_300e_coco.py)       |[model](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20210806_234250-4ff3b67e.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20210806_234250.log.json) |

**Note**:

1. The test score threshold is 0.001.
2. We find that the performance is unstable and may fluctuate by about 0.7 mAP. We will continue to investigate and improve it.
