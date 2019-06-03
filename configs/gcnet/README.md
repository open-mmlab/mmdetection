# GCNet for Object Detection

By [Yue Cao](http://yue-cao.me), [Jiarui Xu](http://jerryxu.net), [Stephen Lin](https://scholar.google.com/citations?user=c3PYmxUAAAAJ&hl=en), Fangyun Wei, [Han Hu](https://sites.google.com/site/hanhushomepage/).

We provide config files to reproduce the results in the paper for
["GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond"](https://arxiv.org/abs/1904.11492) on COCO object detection.

## Introduction

**GCNet** is initially described in [arxiv](https://arxiv.org/abs/1904.11492). Via absorbing advantages of Non-Local Networks (NLNet) and Squeeze-Excitation Networks (SENet),  GCNet provides a simple, fast and effective approach for global context modeling, which generally outperforms both NLNet and SENet on major benchmarks for various recognition tasks.

## Citing GCNet

```
@article{cao2019GCNet,
  title={GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond},
  author={Cao, Yue and Xu, Jiarui and Lin, Stephen and Wei, Fangyun and Hu, Han},
  journal={arXiv preprint arXiv:1904.11492},
  year={2019}
}
```

## Results and models
The results on COCO 2017val are shown in the below table.

| Backbone  | Model            | Context        | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | mask AP | Download  |
| :-------: | :--------------: | :------------: | :-----: | :------: | :-----------------: | :------------: | :----: | :-----: | :-------: |
| R-50-FPN  | Mask             | GC(c3-c5, r16) |   1x    | 4.5      | 0.533              | 10.1           | 38.5   |  35.1   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/gcnet/mask_rcnn_r16_gcb_c3-c5_r50_fpn_1x_20190602-c550c707.pth) |
| R-50-FPN  | Mask             | GC(c3-c5, r4)  |   1x    | 4.6      | 0.533              | 9.9            | 38.9   |  35.5   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/gcnet/mask_rcnn_r4_gcb_c3-c5_r50_fpn_1x_20190602-18ae2dfd.pth) |
| R-101-FPN | Mask             | GC(c3-c5, r16) |   1x    | 7.0      | 0.731              | 8.6            | 40.8   |  37.0   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/gcnet/mask_rcnn_r16_gcb_c3-c5_r101_fpn_1x_20190602-f4456442.pth) |
| R-101-FPN | Mask             | GC(c3-c5, r4)  |   1x    | 7.1      | 0.747              | 8.6            | 40.8   |  36.9   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/gcnet/mask_rcnn_r4_gcb_c3-c5_r101_fpn_1x_20190602-1ee20d5f.pth) |

| Backbone  | Model            | Context        | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | mask AP | Download  |
| :-------: | :--------------: | :------------: | :-----: | :------: | :-----------------: | :------------: | :----: | :-----: | :-------: |
| R-50-FPN  | Mask             |      -         |   1x    | 3.9      | 0.543              | 10.2           | 37.2   |  33.8   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/gcnet/mask_rcnn_r50_fpn_syncbn_1x_20190602-bccc62fa.pth) |
| R-50-FPN  | Mask             | GC(c3-c5, r16) |   1x    | 4.5      | 0.547              | 9.9            | 39.4   |  35.7   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/gcnet/mask_rcnn_r16_gcb_c3-c5_r50_fpn_syncbn_1x_20190602-a0169c20.pth) |
| R-50-FPN  | Mask             | GC(c3-c5, r4)  |   1x    | 4.6      | 0.603              | 9.4            | 39.9   |  36.2   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/gcnet/mask_rcnn_r4_gcb_c3-c5_r50_fpn_syncbn_1x_20190602-ace08792.pth) |
| R-101-FPN | Mask             |      -         |   1x    | 5.8      | 0.665              | 9.2            | 39.8   |  36.0   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/gcnet/mask_rcnn_r101_fpn_syncbn_1x_20190602-b2a0e2b7.pth) |
| R-101-FPN | Mask             | GC(c3-c5, r16) |   1x    | 7.0      | 0.778              | 9.0            | 41.1   |  37.4   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/gcnet/mask_rcnn_r16_gcb_c3-c5_r101_fpn_syncbn_1x_20190602-717e6dbd.pth) |
| R-101-FPN | Mask             | GC(c3-c5, r4)  |   1x    | 7.1      | 0.786              | 8.9            | 41.7   |  37.6   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/gcnet/mask_rcnn_r4_gcb_c3-c5_r101_fpn_syncbn_1x_20190602-a893c718.pth) |
| X-101-FPN | Mask             |      -         |   1x    | 7.1      | 0.912              | 8.5            | 41.2   |  37.3   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/gcnet/mask_rcnn_x101_32x4d_fpn_syncbn_1x_20190602-bb8ae7e5.pth) |
| X-101-FPN | Mask             | GC(c3-c5, r16) |   1x    | 8.2      | 1.055              | 7.7            | 42.4   |  38.0   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/gcnet/mask_rcnn_r16_gcb_c3-c5_x101_32x4d_fpn_syncbn_1x_20190602-c28edb53.pth) |
| X-101-FPN | Mask             | GC(c3-c5, r4)  |   1x    | 8.3      | 1.037              | 7.6            | 42.9   |  38.5   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/gcnet/mask_rcnn_r4_gcb_c3-c5_x101_32x4d_fpn_syncbn_1x_20190602-930b3d51.pth) |
| X-101-FPN | Cascade Mask     |      -         |   1x    | -        | -                  | -              | 44.7   |  38.3   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn_1x_20190602-63a800fb.pth) |
| X-101-FPN | Cascade Mask     | GC(c3-c5, r16) |   1x    | -        | -                  | -              | 45.9   |  39.3   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/gcnet/cascade_mask_rcnn_r16_gcb_c3-c5_x101_32x4d_fpn_syncbn_1x_20190602-3e168d88.pth) |
| X-101-FPN | Cascade Mask     | GC(c3-c5, r4)  |   1x    | -        | -                  | -              | 46.5   |  39.7   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/gcnet/cascade_mask_rcnn_r4_gcb_c3-c5_x101_32x4d_fpn_syncbn_1x_20190602-b579157f.pth) |
| X-101-FPN | DCN Cascade Mask |      -         |   1x    | -        | -                  | -              | 47.1   |  40.4   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/gcnet/cascade_mask_rcnn_dconv_c3-c5_x101_32x4d_fpn_syncbn_1x_20190602-9aa8c394.pth) |
| X-101-FPN | DCN Cascade Mask | GC(c3-c5, r16) |   1x    | -        | -                  | -              | 47.9   |  40.9   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/gcnet/cascade_mask_rcnn_r16_gcb_dconv_c3-c5_x101_32x4d_fpn_syncbn_1x_20190602-b86027a6.pth) |
| X-101-FPN | DCN Cascade Mask | GC(c3-c5, r4)  |   1x    | -        | -                  | -              | 47.9   |  40.8   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/gcnet/cascade_mask_rcnn_r4_gcb_dconv_c3-c5_x101_32x4d_fpn_syncbn_1x_20190602-b4164f6b.pth) |


**Notes:**

- The `SyncBN` is added in the backbone for all models in **Table 2**.
- `GC` denotes Global Context (GC) block is inserted after 1x1 conv of backbone.
- `DCN` denotes replace 3x3 conv with 3x3 Deformable Convolution in `c3-c5` stages of backbone.
- `r4` and `r16` denote ratio 4 and ratio 16 in GC block respectively.

