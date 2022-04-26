# DCNv2

> [Deformable ConvNets v2: More Deformable, Better Results](https://arxiv.org/abs/1811.11168)

<!-- [ALGORITHM] -->

## Abstract

The superior performance of Deformable Convolutional Networks arises from its ability to adapt to the geometric variations of objects. Through an examination of its adaptive behavior, we observe that while the spatial support for its neural features conforms more closely than regular ConvNets to object structure, this support may nevertheless extend well beyond the region of interest, causing features to be influenced by irrelevant image content. To address this problem, we present a reformulation of Deformable ConvNets that improves its ability to focus on pertinent image regions, through increased modeling power and stronger training. The modeling power is enhanced through a more comprehensive integration of deformable convolution within the network, and by introducing a modulation mechanism that expands the scope of deformation modeling. To effectively harness this enriched modeling capability, we guide network training via a proposed feature mimicking scheme that helps the network to learn features that reflect the object focus and classification power of RCNN features. With the proposed contributions, this new version of Deformable ConvNets yields significant performance gains over the original model and produces leading results on the COCO benchmark for object detection and instance segmentation.

## Results and Models

| Backbone         | Model        | Style   | Conv          | Pool   | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
|:----------------:|:------------:|:-------:|:-------------:|:------:|:-------:|:--------:|:--------------:|:------:|:-------:|:------:|:--------:|
| R-50-FPN         | Faster       | pytorch | mdconv(c3-c5) | -      | 1x      | 4.1  | 17.6 | 41.4 |     | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcnv2/faster_rcnn_r50_fpn_mdconv_c3-c5_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_mdconv_c3-c5_1x_coco/faster_rcnn_r50_fpn_mdconv_c3-c5_1x_coco_20200130-d099253b.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_mdconv_c3-c5_1x_coco/faster_rcnn_r50_fpn_mdconv_c3-c5_1x_coco_20200130_222144.log.json) |
| *R-50-FPN (dg=4) | Faster       | pytorch | mdconv(c3-c5) | -      | 1x      | 4.2  | 17.4 | 41.5 |     | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcnv2/faster_rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco/faster_rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco_20200130-01262257.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco/faster_rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco_20200130_222058.log.json) |
| R-50-FPN         | Faster       | pytorch | -             | mdpool | 1x      | 5.8  | 16.6 | 38.7 |     | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcnv2/faster_rcnn_r50_fpn_mdpool_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_mdpool_1x_coco/faster_rcnn_r50_fpn_mdpool_1x_coco_20200307-c0df27ff.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_mdpool_1x_coco/faster_rcnn_r50_fpn_mdpool_1x_coco_20200307_203304.log.json) |
| R-50-FPN         | Mask         | pytorch | mdconv(c3-c5) | -      | 1x      | 4.5  | 15.1 | 41.5 | 37.1 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcnv2/mask_rcnn_r50_fpn_mdconv_c3-c5_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/dcn/mask_rcnn_r50_fpn_mdconv_c3-c5_1x_coco/mask_rcnn_r50_fpn_mdconv_c3-c5_1x_coco_20200203-ad97591f.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/dcn/mask_rcnn_r50_fpn_mdconv_c3-c5_1x_coco/mask_rcnn_r50_fpn_mdconv_c3-c5_1x_coco_20200203_063443.log.json) |
| R-50-FPN (FP16)      | Mask         | pytorch | mdconv(c3-c5)| -      | 1x      | 3.1      |                | 42.0   | 37.6    |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fp16/mask_rcnn_r50_fpn_fp16_mdconv_c3-c5_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/fp16/mask_rcnn_r50_fpn_fp16_mdconv_c3-c5_1x_coco/mask_rcnn_r50_fpn_fp16_mdconv_c3-c5_1x_coco_20210520_180434-cf8fefa5.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/fp16/mask_rcnn_r50_fpn_fp16_mdconv_c3-c5_1x_coco/mask_rcnn_r50_fpn_fp16_mdconv_c3-c5_1x_coco_20210520_180434.log.json)  |

**Notes:**

- `mdconv` denotes modulated deformable convolution, `c3-c5` means adding dconv in resnet stage 3 to 5. `mdpool` denotes modulated deformable roi pooling.
- The dcn ops are modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch, which should be more memory efficient and slightly faster.
- (*) For R-50-FPN (dg=4), dg is short for deformable_group. This model is trained and tested on Amazon EC2 p3dn.24xlarge instance.
- **Memory, Train/Inf time is outdated.**

## Citation

```latex
@article{zhu2018deformable,
  title={Deformable ConvNets v2: More Deformable, Better Results},
  author={Zhu, Xizhou and Hu, Han and Lin, Stephen and Dai, Jifeng},
  journal={arXiv preprint arXiv:1811.11168},
  year={2018}
}
```
