# SCNet

We provide the code for reproducing experiment results of [SCNet](https://arxiv.org/abs/2012.10150).

```
@inproceedings{vu2019cascade,
  title={SCNet: Training Inference Sample Consistency for Instance Segmentation},
  author={Vu, Thang and Haeyong, Kang and Yoo, Chang D},
  booktitle={AAAI},
  year={2021}
}
```

## Benchmark

| Backbone |  Style  | Schedule | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | Mask AP |                   Download                   |
|:--------:|:-------:|:--------:|:--------:|:-------------------:|:--------------:|:------:|:-------:|:--------------------------------------------:|
| R-50-FPN |pytorch  |  20e     |    -     |          -          |        -       |  44.6  |  40.0   | [model](https://drive.google.com/file/d/1X81YEyy7m_yup3Q0tVl8jRhD3RNdEhlz/view?usp=sharing) |
