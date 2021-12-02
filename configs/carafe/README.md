# CARAFE: Content-Aware ReAssembly of FEatures

## Abstract

<!-- [ABSTRACT] -->

Feature upsampling is a key operation in a number of modern convolutional network architectures, e.g. feature pyramids. Its design is critical for dense prediction tasks such as object detection and semantic/instance segmentation. In this work, we propose Content-Aware ReAssembly of FEatures (CARAFE), a universal, lightweight and highly effective operator to fulfill this goal. CARAFE has several appealing properties: (1) Large field of view. Unlike previous works (e.g. bilinear interpolation) that only exploit sub-pixel neighborhood, CARAFE can aggregate contextual information within a large receptive field. (2) Content-aware handling. Instead of using a fixed kernel for all samples (e.g. deconvolution), CARAFE enables instance-specific content-aware handling, which generates adaptive kernels on-the-fly. (3) Lightweight and fast to compute. CARAFE introduces little computational overhead and can be readily integrated into modern network architectures. We conduct comprehensive evaluations on standard benchmarks in object detection, instance/semantic segmentation and inpainting. CARAFE shows consistent and substantial gains across all the tasks (1.2%, 1.3%, 1.8%, 1.1db respectively) with negligible computational overhead. It has great potential to serve as a strong building block for future research. It has great potential to serve as a strong building block for future research.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143872016-48225685-0e59-49cf-bd65-a50ee04ca8a2.png"/>
</div>

<!-- [PAPER_TITLE: CARAFE: Content-Aware ReAssembly of FEatures] -->
<!-- [PAPER_URL: https://arxiv.org/abs/1905.02188] -->

## Citation

<!-- [ALGORITHM] -->

We provide config files to reproduce the object detection & instance segmentation results in the ICCV 2019 Oral paper for [CARAFE: Content-Aware ReAssembly of FEatures](https://arxiv.org/abs/1905.02188).

```
@inproceedings{Wang_2019_ICCV,
    title = {CARAFE: Content-Aware ReAssembly of FEatures},
    author = {Wang, Jiaqi and Chen, Kai and Xu, Rui and Liu, Ziwei and Loy, Chen Change and Lin, Dahua},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2019}
}
```

## Results and Models

The results on COCO 2017 val is shown in the below table.

| Method               | Backbone | Style   | Lr schd | Test Proposal Num | Inf time (fps) | Box AP | Mask AP | Config | Download |
|:--------------------:|:--------:|:-------:|:-------:|:-----------------:|:--------------:|:------:|:-------:|:------:|:--------:|
| Faster R-CNN w/ CARAFE | R-50-FPN | pytorch | 1x      | 1000 | 16.5 | 38.6   | 38.6       | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/carafe/faster_rcnn_r50_fpn_carafe_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/carafe/faster_rcnn_r50_fpn_carafe_1x_coco/faster_rcnn_r50_fpn_carafe_1x_coco_bbox_mAP-0.386_20200504_175733-385a75b7.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/carafe/faster_rcnn_r50_fpn_carafe_1x_coco/faster_rcnn_r50_fpn_carafe_1x_coco_20200504_175733.log.json) |
| -                      |    -     |  -      | -       | 2000 |      |        |            |  |
| Mask R-CNN w/ CARAFE   | R-50-FPN | pytorch | 1x      | 1000 | 14.0 | 39.3   | 35.8       | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/carafe/mask_rcnn_r50_fpn_carafe_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/carafe/mask_rcnn_r50_fpn_carafe_1x_coco/mask_rcnn_r50_fpn_carafe_1x_coco_bbox_mAP-0.393__segm_mAP-0.358_20200503_135957-8687f195.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/carafe/mask_rcnn_r50_fpn_carafe_1x_coco/mask_rcnn_r50_fpn_carafe_1x_coco_20200503_135957.log.json) |
| -                      |   -      |  -      |   -     | 2000 |      |        |            |  |

## Implementation

The CUDA implementation of CARAFE can be find at https://github.com/myownskyW7/CARAFE.
