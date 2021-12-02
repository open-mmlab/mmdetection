# Albu Example

## Abstract

<!-- [ABSTRACT] -->

Data augmentation is a commonly used technique for increasing both the size and the diversity of labeled training sets by leveraging input transformations that preserve output labels. In computer vision domain, image augmentations have become a common implicit regularization technique to combat overfitting in deep convolutional neural networks and are ubiquitously used to improve performance. While most deep learning frameworks implement basic image transformations, the list is typically limited to some variations and combinations of flipping, rotating, scaling, and cropping. Moreover, the image processing speed varies in existing tools for image augmentation. We present Albumentations, a fast and flexible library for image augmentations with many various image transform operations available, that is also an easy-to-use wrapper around other augmentation libraries. We provide examples of image augmentations for different computer vision tasks and show that Albumentations is faster than other commonly used image augmentation tools on the most of commonly used image transformations.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143870703-74f3ea3f-ae23-4035-9856-746bc3f88464.png" height="400" />
</div>

<!-- [PAPER_TITLE: Albumentations: fast and flexible image augmentations] -->
<!-- [PAPER_URL: https://arxiv.org/abs/1809.06839] -->

## Citation

<!-- [OTHERS] -->

```
@article{2018arXiv180906839B,
  author = {A. Buslaev, A. Parinov, E. Khvedchenya, V.~I. Iglovikov and A.~A. Kalinin},
  title = "{Albumentations: fast and flexible image augmentations}",
  journal = {ArXiv e-prints},
  eprint = {1809.06839},
  year = 2018
}
```

## Results and Models

| Backbone  | Style   | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
|:---------:|:-------:|:-------:|:--------:|:--------------:|:------:|:-------:|:------:|:--------:|
| R-50      | pytorch | 1x      | 4.4      | 16.6           |  38.0  | 34.5    |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/albu_example/mask_rcnn_r50_fpn_albu_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/albu_example/mask_rcnn_r50_fpn_albu_1x_coco/mask_rcnn_r50_fpn_albu_1x_coco_20200208-ab203bcd.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/albu_example/mask_rcnn_r50_fpn_albu_1x_coco/mask_rcnn_r50_fpn_albu_1x_coco_20200208_225520.log.json) |
