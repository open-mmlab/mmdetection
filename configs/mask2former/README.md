# Mask2Former

> [Masked-attention Mask Transformer for Universal Image Segmentation](http://arxiv.org/abs/2112.01527)

<!-- [ALGORITHM] -->

## Abstract

Image segmentation is about grouping pixels with different semantics, e.g., category or instance membership, where each choice of semantics defines a task. While only the semantics of each task differ, current research focuses on designing specialized architectures for each task. We present Masked-attention Mask Transformer (Mask2Former), a new architecture capable of addressing any image segmentation task (panoptic, instance or semantic). Its key components include masked attention, which extracts localized features by constraining cross-attention within predicted mask regions. In addition to reducing the research effort by at least three times, it outperforms the best specialized architectures by a significant margin on four popular datasets. Most notably, Mask2Former sets a new state-of-the-art for panoptic segmentation (57.8 PQ on COCO), instance segmentation (50.1 AP on COCO) and semantic segmentation (57.7 mIoU on ADE20K).

<div align=center>
<img src="https://camo.githubusercontent.com/455d3116845b1d580b1f8a8542334b9752fdf39364deee2951cdd231524c7725/68747470733a2f2f626f77656e63303232312e6769746875622e696f2f696d616765732f6d61736b666f726d657276325f7465617365722e706e67" height="300"/>
</div>

## Introduction

Mask2Former requires COCO and [COCO-panoptic](http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip) dataset for training and evaluation. You need to download and extract it in the COCO dataset path.
The directory should be like this.

```none
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
|   |   |   ├── instances_train2017.json
|   |   |   ├── instances_val2017.json
│   │   │   ├── panoptic_train2017.json
│   │   │   ├── panoptic_train2017
│   │   │   ├── panoptic_val2017.json
│   │   │   ├── panoptic_val2017
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```

## Results and Models

| Backbone |  style  |   Pretrain   | Lr schd | Mem (GB) | Inf time (fps) |  PQ   | box mAP | mask mAP |                                                                    Config                                                                     |                                                                                                                                                                                                             Download                                                                                                                                                                                                             |
| :------: | :-----: | :----------: | :-----: | :------: | :------------: | :---: | :-----: | :------: | :-------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-50   | pytorch | ImageNet-1K  |   50e   |   13.9   |       -        | 51.9  |  44.8   |   41.9   |            [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/mask2former/mask2former_r50_lsj_8x2_50e_coco.py)            |                                             [model](https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_r50_lsj_8x2_50e_coco/mask2former_r50_lsj_8x2_50e_coco_20220326_224516-0091ce2b.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_r50_lsj_8x2_50e_coco/mask2former_r50_lsj_8x2_50e_coco_20220326_224516.log.json)                                             |
|  R-101   | pytorch | ImageNet-1K  |   50e   |   16.1   |       -        | 52.4  |  45.3   |   42.4   |           [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/mask2former/mask2former_r101_lsj_8x2_50e_coco.py)            |                                           [model](https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_r101_lsj_8x2_50e_coco/mask2former_r101_lsj_8x2_50e_coco_20220329_225104-bb4df090.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_r101_lsj_8x2_50e_coco/mask2former_r101_lsj_8x2_50e_coco_20220329_225104.log.json)                                           |
|  Swin-T  |    -    | ImageNet-1K  |   50e   |   15.9   |       -        | 53.4  |  46.3   |   43.4   |     [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/mask2former/mask2former_swin-t-p4-w7-224_lsj_8x2_50e_coco.py)      |                   [model](https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-t-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-t-p4-w7-224_lsj_8x2_50e_coco_20220326_224553-c92f921c.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-t-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-t-p4-w7-224_lsj_8x2_50e_coco_20220326_224553.log.json)                   |
|  Swin-S  |    -    | ImageNet-1K  |   50e   |   19.1   |       -        | 54.5  |  47.8   |   44.5   |     [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py)      |                   [model](https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220329_225200-9f633bcf.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220329_225200.log.json)                   |
|  Swin-B  |    -    | ImageNet-1K  |   50e   |   26.0   |       -        | 55.1  |  48.2   |   44.9   |     [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/mask2former/mask2former_swin-b-p4-w12-384_lsj_8x2_50e_coco.py)     |                 [model](https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-b-p4-w12-384_lsj_8x2_50e_coco/mask2former_swin-b-p4-w12-384_lsj_8x2_50e_coco_20220331_002244-1db756b2.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-b-p4-w12-384_lsj_8x2_50e_coco/mask2former_swin-b-p4-w12-384_lsj_8x2_50e_coco_20220331_002244.log.json)                 |
|  Swin-B  |    -    | ImageNet-21K |   50e   |   25.8   |       -        | 56.3  |  50.0   |   46.3   |  [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/mask2former/mask2former_swin-b-p4-w12-384-in21k_lsj_8x2_50e_coco.py)  |     [model](https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-b-p4-w12-384-in21k_lsj_8x2_50e_coco/mask2former_swin-b-p4-w12-384-in21k_lsj_8x2_50e_coco_20220329_230021-89d7c1b1.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-b-p4-w12-384-in21k_lsj_8x2_50e_coco/mask2former_swin-b-p4-w12-384-in21k_lsj_8x2_50e_coco_20220329_230021.log.json)     |
|  Swin-L  |    -    | ImageNet-21K |  100e   |   21.1   |       -        | 57.6  |  52.2   |   48.5   | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/mask2former/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco_20220407_104949-c481ee28.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco_20220407_104949.log.json) |

## Citation

```latex
@article{cheng2021mask2former,
  title={Masked-attention Mask Transformer for Universal Image Segmentation},
  author={Bowen Cheng and Ishan Misra and Alexander G. Schwing and Alexander Kirillov and Rohit Girdhar},
  journal={arXiv},
  year={2021}
}
```
