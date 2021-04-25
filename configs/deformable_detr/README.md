# DETR

## Introduction

[ALGORITHM]

We provide the config files for Deformable DETR: [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159).

```bibtex
@article{zhu2020deformable,
  title={Deformable DETR: Deformable Transformers for End-to-End Object Detection},
  author={Zhu, Xizhou and Su, Weijie and Lu, Lewei and Li, Bin and Wang, Xiaogang and Dai, Jifeng},
  journal={arXiv preprint arXiv:2010.04159},
  year={2020}
}
```

## Results and Models

| Backbone | Model | Lr schd  | box AP | Config | Download |
|:------:|:--------:|:--------------:|:------:|:------:|:--------:|
| R-50 | Deformable DETR  |50e  | 44.5 | [config]() | [model]() &#124; [log]() |
| R-50 | + iterative bounding box refinement  |50e  | 46.1 | [config]() | [model]() &#124; [log]() |
| R-50 | ++ two-stage Deformable DETR  |50e  | 46.8 | [config]() | [model]() &#124; [log]() |
