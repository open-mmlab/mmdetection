# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

## Abstract

<!-- [ABSTRACT] -->

This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision. Challenges in adapting Transformer from language to vision arise from differences between the two domains, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text. To address these differences, we propose a hierarchical Transformer whose representation is computed with Shifted windows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size. These qualities of Swin Transformer make it compatible with a broad range of vision tasks, including image classification (87.3 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as object detection (58.7 box AP and 51.1 mask AP on COCO test-dev) and semantic segmentation (53.5 mIoU on ADE20K val). Its performance surpasses the previous state-of-the-art by a large margin of +2.7 box AP and +2.6 mask AP on COCO, and +3.2 mIoU on ADE20K, demonstrating the potential of Transformer-based models as vision backbones. The hierarchical design and the shifted window approach also prove beneficial for all-MLP architectures.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143999551-6a527048-de38-485c-a1b6-3133ffa5bfaa.png"/>
</div>

<!-- [PAPER_TITLE: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows] -->
<!-- [PAPER_URL: https://arxiv.org/abs/2103.14030] -->

## Citation

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

### Notice
Please follow the example
of `retinanet_swin-t-p4-w7_fpn_1x_coco.py` when you want to combine Swin Transformer with
the one-stage detector. Because there is a layer norm at the outs of Swin Transformer, you must set `start_level` as 0 in FPN, so we have to set the `out_indices` of backbone as `[1,2,3]`.
