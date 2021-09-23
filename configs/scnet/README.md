# SCNet

## Introduction

<!-- [ALGORITHM] -->

We provide the code for reproducing experiment results of [SCNet](https://arxiv.org/abs/2012.10150).

```
@inproceedings{vu2019cascade,
  title={SCNet: Training Inference Sample Consistency for Instance Segmentation},
  author={Vu, Thang and Haeyong, Kang and Yoo, Chang D},
  booktitle={AAAI},
  year={2021}
}
```

## Dataset

SCNet requires COCO and [COCO-stuff](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip) dataset for training. You need to download and extract it in the COCO dataset path.
The directory should be like this.

```none
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
|   |   ├── stuffthingmaps
```

## Results and Models

The results on COCO 2017val are shown in the below table. (results on test-dev are usually slightly higher than val)

|     Backbone    |  Style  | Lr schd | Mem (GB) | Inf speed (fps) | box AP | mask AP | TTA box AP | TTA mask AP | Config |   Download   |
|:---------------:|:-------:|:-------:|:--------:|:---------------:|:------:|:-------:|:----------:|:-----------:|:------:|:------------:|
|     R-50-FPN    | pytorch |    1x   |    7.0   |       6.2       |  43.5  |   39.2  |    44.8    |     40.9    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/scnet/scnet_r50_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/scnet/scnet_r50_fpn_1x_coco/scnet_r50_fpn_1x_coco-c3f09857.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/scnet/scnet_r50_fpn_1x_coco/scnet_r50_fpn_1x_coco_20210117_192725.log.json) |
|     R-50-FPN    | pytorch |   20e   |    7.0   |       6.2       |  44.5  |   40.0  |    45.8    |     41.5    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/scnet/scnet_r50_fpn_20e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/scnet/scnet_r50_fpn_20e_coco/scnet_r50_fpn_20e_coco-a569f645.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/scnet/scnet_r50_fpn_20e_coco/scnet_r50_fpn_20e_coco_20210116_060148.log.json) |
|    R-101-FPN    | pytorch |   20e   |    8.9   |       5.8       |  45.8  |   40.9  |    47.3    |     42.7    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/scnet/scnet_r101_fpn_20e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/scnet/scnet_r101_fpn_20e_coco/scnet_r101_fpn_20e_coco-294e312c.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/scnet/scnet_r101_fpn_20e_coco/scnet_r101_fpn_20e_coco_20210118_175824.log.json) |
| X-101-64x4d-FPN | pytorch |   20e   |   13.2   |       4.9       |  47.5  |   42.3  |    48.9    |     44.0    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/scnet/scnet_x101_64x4d_fpn_20e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/scnet/scnet_x101_64x4d_fpn_20e_coco/scnet_x101_64x4d_fpn_20e_coco-fb09dec9.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/scnet/scnet_x101_64x4d_fpn_20e_coco/scnet_x101_64x4d_fpn_20e_coco_20210120_045959.log.json) |

### Notes

- Training hyper-parameters are identical to those of [HTC](https://github.com/open-mmlab/mmdetection/tree/master/configs/htc).
- TTA means Test Time Augmentation, which applies horizonal flip and multi-scale testing. Refer to [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/scnet/scnet_r50_fpn_1x_coco.py).
