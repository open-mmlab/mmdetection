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

| Backbone        | Batch Size<br>gpus x images\_per\_gpu | Step/Total Epochs | Mem (GB) | Inf time (fps) | box AP | Pytorch Version | Download |
| :-------------: | :--------: |:----------------: | :------: | :------------: | :----: | :-------------: | :------: |
| HourglassNet-104 | [10 x 5](./cornernet_hourglass104_mstest_10x5_210e_coco.py)<br>align paper, need 2 V100 nodes | 180/210 | 13.9 | | 40.6<br>TTA: 41.3 | 1.5 | |
| HourglassNet-104 | [8 x 6](./cornernet_hourglass104_mstest_8x6_210e_coco.py)<br>on single V100 node| 180/210 | 15.9 | | 40.9<br>TTA: 41.2 | 1.5 | |
| HourglassNet-104 | [32 x 3](./cornernet_hourglass104_mstest_32x3_210e_coco.py)<br>for 1080TI, need 4 nodes | 180/210 | 9.5 | | 39.6<br>TTA: 40.4 | 1.3 | |

Note: 
- TTA setting is single-scale and `flip=True`.
- Experiments with `images_per_gpu=6` are implemented on Tesla V100-SXM2-32GB, `images_per_gpu=3` are implemented on GeForce GTX 1080 Ti.
