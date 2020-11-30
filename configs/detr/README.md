# DETR

## Introduction

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
| R-50 | DETR  |150e ||  | | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/detr/detr_r50_8x4_150e_coco.py) | [model]() &#124; [log]() |
