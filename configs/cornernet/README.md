# CornerNet

## Introduction
```
@inproceedings{law2018cornernet,
  title={Cornernet: Detecting objects as paired keypoints},
  author={Law, Hei and Deng, Jia},
  booktitle={15th European Conference on Computer Vision, ECCV 2018},
  pages={765--781},
  year={2018},
  organization={Springer Verlag}
}
```

## Results and models

| Backbone        | Batch Size | Step/Total Epochs | Mem (GB) | Inf time (fps) | box AP | Download |
| :-------------: | :--------: |:----------------: | :------: | :------------: | :----: | :------: |
| HourglassNet-104 | [10 x 5](./cornernet_hourglass104_mstest_10x5_210e_coco.py) | 180/210 | 13.9 | 4.2 | 40.6<br>TTA: 41.3 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection/v2.0/cornernet/cornernet_hourglass104_mstest_10x5_210e_coco/cornernet_hourglass104_mstest_10x5_210e_coco_20200701_022705-4bd2dcbd.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection/v2.0/cornernet/cornernet_hourglass104_mstest_10x5_210e_coco/cornernet_hourglass104_mstest_10x5_210e_coco_20200701_022705.log.json) |
| HourglassNet-104 | [8 x 6](./cornernet_hourglass104_mstest_8x6_210e_coco.py) | 180/210 | 15.9 | 4.2 | 40.9<br>TTA: 41.2 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection/v2.0/cornernet/cornernet_hourglass104_mstest_8x6_210e_coco/cornernet_hourglass104_mstest_8x6_210e_coco_20200701_022705-f419f5d3.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection/v2.0/cornernet/cornernet_hourglass104_mstest_8x6_210e_coco/cornernet_hourglass104_mstest_8x6_210e_coco_20200701_022705.log.json) |
| HourglassNet-104 | [32 x 3](./cornernet_hourglass104_mstest_32x3_210e_coco.py) | 180/210 | 9.5 | 3.9 | 39.6<br>TTA: 40.4 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection/v2.0/cornernet/cornernet_hourglass104_mstest_32x3_210e_coco/cornernet_hourglass104_mstest_32x3_210e_coco_20200614_005449-9b9368d6.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection/v2.0/cornernet/cornernet_hourglass104_mstest_32x3_210e_coco/cornernet_hourglass104_mstest_32x3_210e_coco_20200614_005449.log.json) |

Note:
- TTA setting is single-scale and `flip=True`.
- Experiments with `images_per_gpu=6` are conducted on Tesla V100-SXM2-32GB, `images_per_gpu=3` are conducted on GeForce GTX 1080 Ti.
- Here are the descriptions of each experiment setting:
    - 10 x 5: 10 GPUs with 5 images per gpu. This is the same setting as that reported in the original paper.
    - 8 x 6: 8 GPUs with 6 images per gpu. The total batchsize is similar to paper and only need 1 node to train.
    - 32 x 3: 32 GPUs with 3 images per gpu. The default setting for 1080TI and need 4 nodes to train.
