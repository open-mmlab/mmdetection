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

| Backbone        | Batch Size | Step/Total Epochs | Mem (GB) | Inf time (fps) | box AP | Pytorch Version | Download |
| :-------------: | :--------: |:----------------: | :------: | :------------: | :----: | :-------------: | :------: |
| HourglassNet-104 | 5 x 10 | 180/210 | 13.9 | | 40.6(TTA: 41.3) | 1.5 | |
| HourglassNet-104 | 6 x 8 | 180/210 | 15.9 | | 40.9(TTA: 41.2) | 1.5 | |
| HourglassNet-104 | 6 x 16 | 180/210 | 15.9 | | 40.1(TTA: 41.1) | 1.5 | |
| HourglassNet-104 | 3 x 32 | 180/210 | 9.5 | | 39.6(TTA: 40.4) | 1.3 | |

Note: 
- TTA setting is single-scale and `flip=True`.
- Experiments with `images_per_gpu=6` are implemented on Tesla V100-SXM2-32GB, `images_per_gpu=3` are implemented on GeForce GTX 1080 Ti.
