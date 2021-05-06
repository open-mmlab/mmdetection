# DETR

## Introduction

<!-- [ALGORITHM] -->

We provide the config files for DETR: [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872).

```BibTeX
@inproceedings{detr,
  author    = {Nicolas Carion and
               Francisco Massa and
               Gabriel Synnaeve and
               Nicolas Usunier and
               Alexander Kirillov and
               Sergey Zagoruyko},
  title     = {End-to-End Object Detection with Transformers},
  booktitle = {ECCV},
  year      = {2020}
}
```

## Results and Models

| Backbone | Model | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:------:|:--------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| R-50 | DETR  |150e |7.9|  | 40.1 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/detr/detr_r50_8x2_150e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835.log.json) |
