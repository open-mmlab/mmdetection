
# Cascade RPN

We provide the code for reproducing experiment results of [Cascade RPN](https://arxiv.org/abs/1909.06720). Questions and issues related to Cascade RPN can be discussed in [Official Repo](https://github.com/thangvubk/Cascade-RPN).
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
|   RPN  | R-50-FPN | caffe |     -    |          -          |        -       |   58.3  |                    -                   |
|  CRPN  | R-50-FPN | caffe |     -    |          -          |        -       |   71.7  | [model](http://bit.ly/cascade_rpn_r50) |

### Detection performance
|     Method    |   Proposal  | Backbone |  Style  | Schedule | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP |                   Download                   |
|:-------------:|:-----------:|:--------:|:-------:|:--------:|:--------:|:-------------------:|:--------------:|:------:|:--------------------------------------------:|
|   Fast R-CNN  |     RPN     | R-50-FPN |  caffe  |    1x    |    3.5   |        0.250        |      16.5      |  36.9  |                       -                      |
|   Fast R-CNN  | Cascade RPN | R-50-FPN |  caffe  |    1x    |    3.5   |        0.250        |      16.5      |  40.0  |     [model](http://bit.ly/crpn_fast_r50)     |
|  Faster R-CNN |     RPN     | R-50-FPN |  caffe  |    1x    |    3.8   |        0.353        |      13.6      |  37.0  |                       -                      |
|  Faster R-CNN | Cascade RPN | R-50-FPN |  caffe  |    1x    |    4.6   |        0.561        |      11.1      |  40.5  |    [model](http://bit.ly/crpn_faster_r50)    |
| Cascade R-CNN |     RPN     | R-50-FPN | pytorch |    1x    |    4.1   |        0.455        |      11.9      |  40.8  |                       -                      |
| Cascade R-CNN | Cascade RPN | R-50-FPN | pytorch |    1x    |    5.2   |        0.650        |       9.6      |  41.6  | [model](http://bit.ly/crpn_cascade_rcnn_r50) |
