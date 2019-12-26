# Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection


## Introduction

```
@article{zhang2019bridging,
  title   =  {Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection},
  author  =  {Zhang, Shifeng and Chi, Cheng and Yao, Yongqiang and Lei, Zhen and Li, Stan Z.},
  journal =  {arXiv preprint arXiv:1912.02424},
  year    =  {2019}
}
```


## Results and Models

| Backbone  | Style   | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP |
|:---------:|:-------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|
| R-50      | pytorch | 1x      |          |                     |                |  38.8  |


**Notes:**
- Due to the hardware limit, we train the model use 2 GPUs and set `lr = 0.01 / 4`