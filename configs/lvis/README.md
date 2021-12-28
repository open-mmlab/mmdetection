# LVIS: A Dataset for Large Vocabulary Instance Segmentation

## Abstract

<!-- [ABSTRACT] -->

Progress on object detection is enabled by datasets that focus the research community's attention on open challenges. This process led us from simple images to complex scenes and from bounding boxes to segmentation masks. In this work, we introduce LVIS (pronounced `el-vis'): a new dataset for Large Vocabulary Instance Segmentation. We plan to collect ~2 million high-quality instance segmentation masks for over 1000 entry-level object categories in 164k images. Due to the Zipfian distribution of categories in natural images, LVIS naturally has a long tail of categories with few training samples. Given that state-of-the-art deep learning methods for object detection perform poorly in the low-sample regime, we believe that our dataset poses an important and exciting new scientific challenge.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143967423-85b9b705-05ea-4bbc-9a41-eccc14240c7a.png" height="300"/>
</div>

<!-- [PAPER_TITLE: LVIS: A Dataset for Large Vocabulary Instance Segmentation] -->
<!-- [PAPER_URL: https://arxiv.org/abs/1908.03195] -->

## Citation

<!-- [DATASET] -->

```latex
@inproceedings{gupta2019lvis,
  title={{LVIS}: A Dataset for Large Vocabulary Instance Segmentation},
  author={Gupta, Agrim and Dollar, Piotr and Girshick, Ross},
  booktitle={Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

## Common Setting

* Please follow [install guide](../../docs/get_started.md#install-mmdetection) to install open-mmlab forked cocoapi first.
* Run following scripts to install our forked lvis-api.

    ```shell
    pip install git+https://github.com/lvis-dataset/lvis-api.git
    ```

* All experiments use oversample strategy [here](../../docs/tutorials/customize_dataset.md#class-balanced-dataset) with oversample threshold `1e-3`.
* The size of LVIS v0.5 is half of COCO, so schedule `2x` in LVIS is roughly the same iterations as `1x` in COCO.

## Results and models of LVIS v0.5

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: |:--------: |
|    R-50-FPN     | pytorch |   2x    | -        | -              | 26.1   | 25.9    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis_v0.5.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis-dbd06831.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis_20200531_160435.log.json)  |
|    R-101-FPN    | pytorch |   2x    | -        | -              | 27.1   | 27.0    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_2x_lvis_v0.5.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_2x_lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_2x_lvis-54582ee2.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_2x_lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_2x_lvis_20200601_134748.log.json)  |
| X-101-32x4d-FPN | pytorch |   2x    | -        | -              | 26.7   | 26.9    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_2x_lvis_v0.5.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_2x_lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_2x_lvis-3cf55ea2.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_2x_lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_2x_lvis_20200531_221749.log.json)  |
| X-101-64x4d-FPN | pytorch |   2x    | -        |   -            | 26.4   | 26.0    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_2x_lvis_v0.5.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_2x_lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_2x_lvis-1c99a5ad.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_2x_lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_2x_lvis_20200601_194651.log.json)  |

## Results and models of LVIS v1

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: | :--------: |
|    R-50-FPN     | pytorch |   1x    | 9.1      | -              | 22.5   | 21.7    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1-aa78ac3d.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1-20200829_061305.log.json)  |
|    R-101-FPN    | pytorch |   1x    | 10.8     | -              | 24.6   | 23.6    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_1x_lvis_v1.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_r101_fpn_sample1e-3_mstrain_1x_lvis_v1-ec55ce32.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_r101_fpn_sample1e-3_mstrain_1x_lvis_v1-20200829_070959.log.json)  |
| X-101-32x4d-FPN | pytorch |   1x    | 11.8     | -              | 26.7   | 25.5    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_1x_lvis_v1.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_1x_lvis_v1-ebbc5c81.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_1x_lvis_v1-20200829_071317.log.json)  |
| X-101-64x4d-FPN | pytorch |   1x    | 14.6     | -              | 27.2   | 25.8    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_1x_lvis_v1.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_1x_lvis_v1-43d9edfe.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_1x_lvis_v1-20200830_060206.log.json)  |
