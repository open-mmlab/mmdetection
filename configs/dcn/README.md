# Deformable Convolutional Networks

# Introduction

```
@inproceedings{dai2017deformable,
  title={Deformable Convolutional Networks},
  author={Dai, Jifeng and Qi, Haozhi and Xiong, Yuwen and Li, Yi and Zhang, Guodong and Hu, Han and Wei, Yichen},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  year={2017}
}

@article{zhu2018deformable,
  title={Deformable ConvNets v2: More Deformable, Better Results},
  author={Zhu, Xizhou and Hu, Han and Lin, Stephen and Dai, Jifeng},
  journal={arXiv preprint arXiv:1811.11168},
  year={2018}
}
```

## Results and Models

| Backbone  | Model        | Style   | Conv          | Pool   | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | mask AP | Download |
|:---------:|:------------:|:-------:|:-------------:|:------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:-------:|:--------:|
| R-50-FPN  | Faster       | pytorch | dconv(c3-c5)  | -      | 1x      | 3.9      | 0.594               | 10.2           | 40.0   | -       | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/faster_rcnn_dconv_c3-c5_r50_fpn_1x_20190125-e41688c9.pth) |
| R-50-FPN  | Faster       | pytorch | mdconv(c3-c5) | -      | 1x      | 3.7      | 0.598               | 10.0           | 40.2   | -       | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/faster_rcnn_mdconv_c3-c5_r50_fpn_1x_20190125-1b768045.pth) |
| R-50-FPN  | Faster       | pytorch | -             | dpool  | 1x      | 4.6      | 0.714               | 8.7            | 37.8   | -       | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/faster_rcnn_dpool_r50_fpn_1x_20190125-f4fc1d70.pth) |
| R-50-FPN  | Faster       | pytorch | -             | mdpool | 1x      | 5.2      | 0.769               | 8.2            | 38.0   | -       | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/faster_rcnn_mdpool_r50_fpn_1x_20190125-473d0f3d.pth) |
| R-101-FPN | Faster       | pytorch | dconv(c3-c5)  | -      | 1x      | 5.8      | 0.811               | 8.0            | 42.1   | -       | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/faster_rcnn_dconv_c3-c5_r101_fpn_1x_20190125-a7e31b65.pth) |
| X-101-32x4d-FPN | Faster       | pytorch | dconv(c3-c5)  | -      | 1x      | 7.1      | 1.126               | 6.6            | 43.4   | -       | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/faster_rcnn_dconv_c3-c5_x101_32x4d_fpn_1x_20190201-6d46376f.pth) |
| R-50-FPN  | Mask         | pytorch | dconv(c3-c5)  | -      | 1x      | 4.5      | 0.712               | 7.7            | 41.1   | 37.2    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/mask_rcnn_dconv_c3-c5_r50_fpn_1x_20190125-4f94ff79.pth) |
| R-50-FPN  | Mask         | pytorch | mdconv(c3-c5) | -      | 1x      | 4.5      | 0.712               | 7.7            | 41.3   | 37.3    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/mask_rcnn_mdconv_c3-c5_r50_fpn_1x_20190125-c5601dc3.pth) |
| R-101-FPN | Mask         | pytorch | dconv(c3-c5)  | -      | 1x      | 6.4      | 0.939               | 6.5            | 43.2   | 38.7    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/mask_rcnn_dconv_c3-c5_r101_fpn_1x_20190125-decb6db5.pth) |
| R-50-FPN  | Cascade      | pytorch | dconv(c3-c5)  | -      | 1x      | 4.4      | 0.660               | 7.6            | 44.0   | -       | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_20190125-dfa53166.pth) |
| R-101-FPN | Cascade      | pytorch | dconv(c3-c5)  | -      | 1x      | 6.3      | 0.881               | 6.8            | 45.0   | -       | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/cascade_rcnn_dconv_c3-c5_r101_fpn_1x_20190125-aaa877cc.pth) |
| R-50-FPN  | Cascade Mask | pytorch | dconv(c3-c5)  | -      | 1x      | 6.6      | 0.942               | 5.7            | 44.4   | 38.3    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/cascade_mask_rcnn_dconv_c3-c5_r50_fpn_1x_20190125-09d8a443.pth) |
| R-101-FPN | Cascade Mask | pytorch | dconv(c3-c5)  | -      | 1x      | 8.5      | 1.156               | 5.1            | 45.7   | 39.4    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/cascade_mask_rcnn_dconv_c3-c5_r101_fpn_1x_20190125-0d62c190.pth) |

**Notes:**

- `dconv` and `mdconv` denote (modulated) deformable convolution, `c3-c5` means adding dconv in resnet stage 3 to 5. `dpool` and `mdpool` denote (modulated) deformable roi pooling.
- The dcn ops are modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch, which should be more memory efficient and slightly faster.
- **Memory, Train/Inf time is outdated.**