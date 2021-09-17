# Cascade RPN

<!-- [ALGORITHM] -->

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

| Method | Backbone | Style | Mem (GB) | Train time (s/iter) | Inf time (fps) | AR 1000 | Config |               Download                |
|:------:|:--------:|:-----:|:--------:|:-------------------:|:--------------:|:-------:|:-------:|:--------------------------------------:|
|  CRPN  | R-50-FPN | caffe |     -    |          -          |        -       |   72.0  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rpn/crpn_r50_caffe_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/cascade_rpn/crpn_r50_caffe_fpn_1x_coco/cascade_rpn_r50_caffe_fpn_1x_coco-7aa93cef.pth) |

### Detection performance

|     Method    |   Proposal  | Backbone |  Style  | Schedule | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | Config |           Download                   |
|:-------------:|:-----------:|:--------:|:-------:|:--------:|:--------:|:-------------------:|:--------------:|:------:|:-------:|:--------------------------------------------:|
|   Fast R-CNN  | Cascade RPN | R-50-FPN |  caffe  |    1x    |    -     |          -          |        -       |  39.9  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rpn/crpn_fast_rcnn_r50_caffe_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/cascade_rpn/crpn_fast_rcnn_r50_caffe_fpn_1x_coco/crpn_fast_rcnn_r50_caffe_fpn_1x_coco-cb486e66.pth) |
|  Faster R-CNN | Cascade RPN | R-50-FPN |  caffe  |    1x    |    -     |          -          |        -       |  40.4  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rpn/crpn_faster_rcnn_r50_caffe_fpn_1x_coco.py) |[model](https://download.openmmlab.com/mmdetection/v2.0/cascade_rpn/crpn_faster_rcnn_r50_caffe_fpn_1x_coco/crpn_faster_rcnn_r50_caffe_fpn_1x_coco-c8283cca.pth) |
