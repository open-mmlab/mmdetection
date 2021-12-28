# Feature Pyramid Grids
<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Feature pyramid networks have been widely adopted in the object detection literature to improve feature representations for better handling of variations in scale. In this paper, we present Feature Pyramid Grids (FPG), a deep multi-pathway feature pyramid, that represents the feature scale-space as a regular grid of parallel bottom-up pathways which are fused by multi-directional lateral connections. FPG can improve single-pathway feature pyramid networks by significantly increasing its performance at similar computation cost, highlighting importance of deep pyramid representations. In addition to its general and uniform structure, over complicated structures that have been found with neural architecture search, it also compares favorably against such approaches without relying on search. We hope that FPG with its uniform and effective nature can serve as a strong component for future work in object recognition.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143885611-85902399-2885-4a85-9126-9b9b7464ad08.png"/>
</div>

<!-- [PAPER_TITLE: Feature Pyramid Grids] -->
<!-- [PAPER_URL: https://arxiv.org/abs/2004.03580] -->

## Citation

```latex
@article{chen2020feature,
  title={Feature pyramid grids},
  author={Chen, Kai and Cao, Yuhang and Loy, Chen Change and Lin, Dahua and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2004.03580},
  year={2020}
}
```

## Results and Models

We benchmark the new training schedule (crop training, large batch, unfrozen BN, 50 epochs) introduced in NAS-FPN.
All backbones are Resnet-50 in pytorch style.

| Method       | Neck        | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
|:------------:|:-----------:|:-------:|:--------:|:--------------:|:------:|:-------:|:-------:|:--------:|
| Faster R-CNN | FPG         | 50e     | 20.0     | -              | 42.2   | -       |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fpg/faster_rcnn_r50_fpg_crop640_50e_coco.py) |[model](https://download.openmmlab.com/mmdetection/v2.0/fpg/faster_rcnn_r50_fpg_crop640_50e_coco/faster_rcnn_r50_fpg_crop640_50e_coco-76220505.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/fpg/faster_rcnn_r50_fpg_crop640_50e_coco/20210218_223520.log.json) |
| Faster R-CNN | FPG-chn128  | 50e     | 11.9     | -              | 41.2   | -       |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fpg/faster_rcnn_r50_fpg-chn128_crop640_50e_coco.py) |[model](https://download.openmmlab.com/mmdetection/v2.0/fpg/faster_rcnn_r50_fpg-chn128_crop640_50e_coco/faster_rcnn_r50_fpg-chn128_crop640_50e_coco-24257de9.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/fpg/faster_rcnn_r50_fpg-chn128_crop640_50e_coco/20210218_221412.log.json) |
| Mask R-CNN   | FPG         | 50e     | 23.2     | -              | 42.7   | 37.8    |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fpg/mask_rcnn_r50_fpg_crop640_50e_coco.py) |[model](https://download.openmmlab.com/mmdetection/v2.0/fpg/mask_rcnn_r50_fpg_crop640_50e_coco/mask_rcnn_r50_fpg_crop640_50e_coco-c5860453.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/fpg/mask_rcnn_r50_fpg_crop640_50e_coco/20210222_205447.log.json) |
| Mask R-CNN   | FPG-chn128  | 50e     | 15.3     | -              | 41.7   | 36.9    |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fpg/mask_rcnn_r50_fpg-chn128_crop640_50e_coco.py) |[model](https://download.openmmlab.com/mmdetection/v2.0/fpg/mask_rcnn_r50_fpg-chn128_crop640_50e_coco/mask_rcnn_r50_fpg-chn128_crop640_50e_coco-5c6ea10d.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/fpg/mask_rcnn_r50_fpg-chn128_crop640_50e_coco/20210223_025039.log.json) |
| RetinaNet    | FPG         | 50e     | 20.8     | -              | 40.5   | -       |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fpg/retinanet_r50_fpg_crop640_50e_coco.py) |[model](https://download.openmmlab.com/mmdetection/v2.0/fpg/retinanet_r50_fpg_crop640_50e_coco/retinanet_r50_fpg_crop640_50e_coco-46fdd1c6.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/fpg/retinanet_r50_fpg_crop640_50e_coco/20210225_143957.log.json) |
| RetinaNet    | FPG-chn128  | 50e     | 19.9     | -              | 40.3   | -       |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fpg/retinanet_r50_fpg-chn128_crop640_50e_coco.py) |[model](https://download.openmmlab.com/mmdetection/v2.0/fpg/retinanet_r50_fpg-chn128_crop640_50e_coco/retinanet_r50_fpg-chn128_crop640_50e_coco-5cf33c76.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/fpg/retinanet_r50_fpg-chn128_crop640_50e_coco/20210225_184328.log.json) |

**Note**: Chn128 means to decrease the number of channels of features and convs from 256 (default) to 128 in
Neck and BBox Head, which can greatly decrease memory consumption without sacrificing much precision.
