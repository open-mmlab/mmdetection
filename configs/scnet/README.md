# SCNet

> [SCNet: Training Inference Sample Consistency for Instance Segmentation](https://arxiv.org/abs/2012.10150)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Cascaded architectures have brought significant performance improvement in object detection and instance segmentation. However, there are lingering issues regarding the disparity in the Intersection-over-Union (IoU) distribution of the samples between training and inference. This disparity can potentially exacerbate detection accuracy. This paper proposes an architecture referred to as Sample Consistency Network (SCNet) to ensure that the IoU distribution of the samples at training time is close to that at inference time. Furthermore, SCNet incorporates feature relay and utilizes global contextual information to further reinforce the reciprocal relationships among classifying, detecting, and segmenting sub-tasks. Extensive experiments on the standard COCO dataset reveal the effectiveness of the proposed method over multiple evaluation metrics, including box AP, mask AP, and inference speed. In particular, while running 38% faster, the proposed SCNet improves the AP of the box and mask predictions by respectively 1.3 and 2.3 points compared to the strong Cascade Mask R-CNN baseline.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143974840-8fed75f3-661e-4e2a-a210-acf4ab5f42a3.png"/>
</div>

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

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf speed (fps) | box AP | mask AP | TTA box AP | TTA mask AP |                    Config                    |                                                                                                                                           Download                                                                                                                                           |
| :-------------: | :-----: | :-----: | :------: | :-------------: | :----: | :-----: | :--------: | :---------: | :------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|    R-50-FPN     | pytorch |   1x    |   7.0    |       6.2       |  43.5  |  39.2   |    44.8    |    40.9     |     [config](./scnet_r50_fpn_1x_coco.py)     |                 [model](https://download.openmmlab.com/mmdetection/v2.0/scnet/scnet_r50_fpn_1x_coco/scnet_r50_fpn_1x_coco-c3f09857.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/scnet/scnet_r50_fpn_1x_coco/scnet_r50_fpn_1x_coco_20210117_192725.log.json)                 |
|    R-50-FPN     | pytorch |   20e   |   7.0    |       6.2       |  44.5  |  40.0   |    45.8    |    41.5     |    [config](./scnet_r50_fpn_20e_coco.py)     |               [model](https://download.openmmlab.com/mmdetection/v2.0/scnet/scnet_r50_fpn_20e_coco/scnet_r50_fpn_20e_coco-a569f645.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/scnet/scnet_r50_fpn_20e_coco/scnet_r50_fpn_20e_coco_20210116_060148.log.json)               |
|    R-101-FPN    | pytorch |   20e   |   8.9    |       5.8       |  45.8  |  40.9   |    47.3    |    42.7     |    [config](./scnet_r101_fpn_20e_coco.py)    |             [model](https://download.openmmlab.com/mmdetection/v2.0/scnet/scnet_r101_fpn_20e_coco/scnet_r101_fpn_20e_coco-294e312c.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/scnet/scnet_r101_fpn_20e_coco/scnet_r101_fpn_20e_coco_20210118_175824.log.json)             |
| X-101-64x4d-FPN | pytorch |   20e   |   13.2   |       4.9       |  47.5  |  42.3   |    48.9    |    44.0     | [config](./scnet_x101-64x4d_fpn_20e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/scnet/scnet_x101_64x4d_fpn_20e_coco/scnet_x101_64x4d_fpn_20e_coco-fb09dec9.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/scnet/scnet_x101_64x4d_fpn_20e_coco/scnet_x101_64x4d_fpn_20e_coco_20210120_045959.log.json) |

### Notes

- Training hyper-parameters are identical to those of [HTC](https://github.com/open-mmlab/mmdetection/tree/main/configs/htc).
- TTA means Test Time Augmentation, which applies horizontal flip and multi-scale testing. Refer to [config](./scnet_r50_fpn_1x_coco.py).

## Citation

We provide the code for reproducing experiment results of [SCNet](https://arxiv.org/abs/2012.10150).

```latex
@inproceedings{vu2019cascade,
  title={SCNet: Training Inference Sample Consistency for Instance Segmentation},
  author={Vu, Thang and Haeyong, Kang and Yoo, Chang D},
  booktitle={AAAI},
  year={2021}
}
```
