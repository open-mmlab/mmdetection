# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{liu2021Swin,
    title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
    author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
    journal={arXiv preprint arXiv:2103.14030},
    year={2021}
}
```

## Results and models

### Mask R-CNN

| Backbone | Pretrain    | Lr schd | Multi-scale crop     |   FP16   |Mem (GB) | Inf time (fps) | box AP | mask AP |  Config  |   Download  |
| :------: | :---------: | :-----: | :-------------------:| :------: |:------: | :------------: | :----: | :-----: | :------: |  :--------: |
|  Swin-T  | ImageNet-1K |    1x   |        no            |    no    |   7.6   |                |  42.7  |  39.3   | [config](./mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py)             | [model](https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth)  &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937.log.json) |
|  Swin-T  | ImageNet-1K |    3x   |        yes           |    no    |   10.2  |                |  46.0  |  41.6   | [config](./mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py)     | [model](https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_20210906_131725-bacf6f7b.pth)  &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_20210906_131725.log.json) |
|  Swin-T  | ImageNet-1K |    3x   |        yes           |    yes   |   7.8   |                |  46.0  |  41.7   | [config](./mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py)| [model](https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth)  &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006.log.json) |
|  Swin-S  | ImageNet-1K |    3x   |        yes           |    yes   |   11.9  |                |  48.2  |  43.2   | [config](./mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py)| [model](https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth)  &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808.log.json) |
