# An Empirical Study of Spatial Attention Mechanisms in Deep Networks

## Introduction

```
@article{zhu2019empirical,
  title={An Empirical Study of Spatial Attention Mechanisms in Deep Networks},
  author={Zhu, Xizhou and Cheng, Dazhi and Zhang, Zheng and Lin, Stephen and Dai, Jifeng},
  journal={arXiv preprint arXiv:1904.05873},
  year={2019}
}
```


## Results and Models

| Backbone  | Attention Component | DCN  | Lr schd | box AP | Download |
|:---------:|:-------------------:|:----:|:-------:|:------:|:--------:|
| R-50      | 1111                | N    | 1x      | 38.6   |     -    |
| R-50      | 0010                | N    | 1x      | 38.2   |     -    |
| R-50      | 1111                | Y    | 1x      | 41.0   |     -    |
| R-50      | 0010                | Y    | 1x      | 40.8   |     -    |

