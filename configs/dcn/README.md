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

| Backbone         | Model        | Style   | Conv          | Pool   | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Download |
|:----------------:|:------------:|:-------:|:-------------:|:------:|:-------:|:--------:|:--------------:|:------:|:-------:|:--------:|
| R-50-FPN         | Faster       | pytorch | dconv(c3-c5)  | -      | 1x      | 4.0  | 17.8 | 41.3 |     | [model](http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130_212941.log.json) |
| R-50-FPN         | Faster       | pytorch | mdconv(c3-c5) | -      | 1x      | 4.1  | 17.6 | 41.4 |     | [model](http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_mdconv_c3-c5_1x_coco/faster_rcnn_r50_fpn_mdconv_c3-c5_1x_coco_20200130-d099253b.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_mdconv_c3-c5_1x_coco/faster_rcnn_r50_fpn_mdconv_c3-c5_1x_coco_20200130_222144.log.json) |
| *R-50-FPN (dg=4) | Faster       | pytorch | mdconv(c3-c5) | -      | 1x      | 4.2  | 17.4 | 41.5 |     | [model](http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco/faster_rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco_20200130-01262257.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco/faster_rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco_20200130_222058.log.json) |
| R-50-FPN         | Faster       | pytorch | -             | dpool  | 1x      | 5.0  | 17.2 | 38.9 |     | [model](http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_dpool_1x_coco/faster_rcnn_r50_fpn_dpool_1x_coco_20200307-90d3c01d.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_dpool_1x_coco/faster_rcnn_r50_fpn_dpool_1x_coco_20200307_203250.log.json) |
| R-50-FPN         | Faster       | pytorch | -             | mdpool | 1x      | 5.8  | 16.6 | 38.7 |     | [model](http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_mdpool_1x_coco/faster_rcnn_r50_fpn_mdpool_1x_coco_20200307-c0df27ff.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_mdpool_1x_coco/faster_rcnn_r50_fpn_mdpool_1x_coco_20200307_203304.log.json) |
| R-101-FPN        | Faster       | pytorch | dconv(c3-c5)  | -      | 1x      | 6.0  | 12.5 | 42.7 |     | [model](http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r101_fpn_dconv_c3-c5_1x_coco/faster_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203-1377f13d.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r101_fpn_dconv_c3-c5_1x_coco/faster_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203_230019.log.json) |
| X-101-32x4d-FPN | Faster        | pytorch | dconv(c3-c5)  | -      | 1x      | 7.3  | 10.0  | 44.5 |     | [model](http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco/faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco_20200203-4f85c69c.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco/faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco_20200203_001325.log.json) |
| R-50-FPN         | Mask         | pytorch | dconv(c3-c5)  | -      | 1x      | 4.5  | 15.4 | 41.8 | 37.4 | [model](http://download.openmmlab.com/mmdetection/v2.0/dcn/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200203-4d9ad43b.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dcn/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200203_061339.log.json) |
| R-50-FPN         | Mask         | pytorch | mdconv(c3-c5) | -      | 1x      | 4.5  | 15.1 | 41.5 | 37.1 | [model](http://download.openmmlab.com/mmdetection/v2.0/dcn/mask_rcnn_r50_fpn_mdconv_c3-c5_1x_coco/mask_rcnn_r50_fpn_mdconv_c3-c5_1x_coco_20200203-ad97591f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dcn/mask_rcnn_r50_fpn_mdconv_c3-c5_1x_coco/mask_rcnn_r50_fpn_mdconv_c3-c5_1x_coco_20200203_063443.log.json) |
| R-101-FPN        | Mask         | pytorch | dconv(c3-c5)  | -      | 1x      | 6.5  | 11.7  | 43.5 | 38.9  | [model](http://download.openmmlab.com/mmdetection/v2.0/dcn/mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco/mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200216-a71f5bce.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dcn/mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco/mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200216_191601.log.json) |
| R-50-FPN         | Cascade      | pytorch | dconv(c3-c5)  | -      | 1x      | 4.5  | 14.6 | 43.8 |     | [model](http://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-2f1fca44.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130_220843.log.json) |
| R-101-FPN        | Cascade      | pytorch | dconv(c3-c5)  | -      | 1x      | 6.4  | 11.0 | 45.0 |     | [model](http://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203-3b2f0594.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203_224829.log.json) |
| R-50-FPN         | Cascade Mask | pytorch | dconv(c3-c5)  | -      | 1x      | 6.0  | 10.0  | 44.4 | 38.6 | [model](http://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco/cascade_mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200202-42e767a2.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco/cascade_mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200202_010309.log.json) |
| R-101-FPN        | Cascade Mask | pytorch | dconv(c3-c5)  | -      | 1x      | 8.0  | 8.6  | 45.8 | 39.7 | [model](http://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco/cascade_mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200204-df0c5f10.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco/cascade_mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200204_134006.log.json) |
| X-101-32x4d-FPN        | Cascade Mask | pytorch | dconv(c3-c5)  | -      | 1x      | 9.2 |   | 47.3 | 41.1 | [model](http://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco-e75f90c8.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco-20200606_183737.log.json) |

**Notes:**

- `dconv` and `mdconv` denote (modulated) deformable convolution, `c3-c5` means adding dconv in resnet stage 3 to 5. `dpool` and `mdpool` denote (modulated) deformable roi pooling.
- The dcn ops are modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch, which should be more memory efficient and slightly faster.
- (*) For R-50-FPN (dg=4), dg is short for deformable_group. This model is trained and tested on Amazon EC2 p3dn.24xlarge instance.
- **Memory, Train/Inf time is outdated.**
