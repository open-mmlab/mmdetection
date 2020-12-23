# Cascade RPN

We provide the code for reproducing experiment results of [Cascade RPN](https://arxiv.org/abs/1909.06720).

```
@inproceedings{vu2019cascade,
  title={Cascade RPN: Delving into High-Quality Region Proposal Network with Adaptive Convolution},
  author={Vu, Thang and Jang, Hyunjun and Pham, Trung X and Yoo, Chang D},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
  year={2019}
}
```

## Benchmark

### Region proposal performance

| Method | Backbone | Style | Mem (GB) | Train time (s/iter) | Inf time (fps) | AR 1000 |                Download                |
|:------:|:--------:|:-----:|:--------:|:-------------------:|:--------------:|:-------:|:--------------------------------------:|
|  CRPN  | R-50-FPN | caffe |     -    |          -          |        -       |   72.0  | [model](https://drive.google.com/file/d/1qxVdOnCgK-ee7_z0x6mvAir_glMu2Ihi/view?usp=sharing) |

### Detection performance

|     Method    |   Proposal  | Backbone |  Style  | Schedule | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP |                   Download                   |
|:-------------:|:-----------:|:--------:|:-------:|:--------:|:--------:|:-------------------:|:--------------:|:------:|:--------------------------------------------:|
|   Fast R-CNN  | Cascade RPN | R-50-FPN |  caffe  |    1x    |    -     |          -          |        -       |  39.9  | [model](https://drive.google.com/file/d/1NmbnuY5VHi8I9FE8xnp5uNvh2i-t-6_L/view?usp=sharing) |
|  Faster R-CNN | Cascade RPN | R-50-FPN |  caffe  |    1x    |    -     |          -          |        -       |  40.4  | [model](https://drive.google.com/file/d/1dS3Q66qXMJpcuuQgDNkLp669E5w1UMuZ/view?usp=sharing) |
