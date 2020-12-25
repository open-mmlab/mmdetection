# Sparse R-CNN: End-to-End Object Detection with Learnable Proposals

## Introduction

```
@article{peize2020sparse,
  title   =  {{SparseR-CNN}: End-to-End Object Detection with Learnable Proposals},
  author  =  {Peize Sun and Rufeng Zhang and Yi Jiang and Tao Kong and Chenfeng Xu and Wei Zhan and Masayoshi Tomizuka and Lei Li and Zehuan Yuan and Changhu Wang and Ping Luo},
  journal =  {arXiv preprint arXiv:2011.12450},
  year    =  {2020}
}
```

## Results and Models

| Model        | Backbone  | Style   | Lr schd | Number of Proposals |Multi-Scale| RandomCrop  | box AP  | Config | Download |
|:------------:|:---------:|:-------:|:-------:|:-------:|:-------: |:---------:|:------:|:------:|:--------:|
| Sparse R-CNN | R-50-FPN  | pytorch | 3x      | 3x      |   100   | False     |  42.8  |         |       |
| Sparse R-CNN | R-50-FPN  | pytorch | 3x      | 3x      |   300   | True      |  45.0  |         |       |

### Notes:
We observe about 0.3 AP noise.
