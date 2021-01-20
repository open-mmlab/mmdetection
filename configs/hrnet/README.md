# High-resolution networks (HRNets) for object detection

## Introduction

[BACKBONE]

```latex
@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{SunZJCXLMWLW19,
  title={High-Resolution Representations for Labeling Pixels and Regions},
  author={Ke Sun and Yang Zhao and Borui Jiang and Tianheng Cheng and Bin Xiao
  and Dong Liu and Yadong Mu and Xinggang Wang and Wenyu Liu and Jingdong Wang},
  journal   = {CoRR},
  volume    = {abs/1904.04514},
  year={2019}
}
```

## Results and Models

### Faster R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :-------------:|:------:| :------:| :--------:|
|   HRNetV2p-W18  | pytorch |   1x    | 6.6      | 13.4           | 36.9   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/faster_rcnn_hrnetv2p_w18_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w18_1x_coco/faster_rcnn_hrnetv2p_w18_1x_coco_20200130-56651a6d.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w18_1x_coco/faster_rcnn_hrnetv2p_w18_1x_coco_20200130_211246.log.json) |
|   HRNetV2p-W18  | pytorch |   2x    | 6.6      |                | 38.9   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/faster_rcnn_hrnetv2p_w18_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w18_2x_coco/faster_rcnn_hrnetv2p_w18_2x_coco_20200702_085731-a4ec0611.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w18_2x_coco/faster_rcnn_hrnetv2p_w18_2x_coco_20200702_085731.log.json) |
|   HRNetV2p-W32  | pytorch |   1x    | 9.0      | 12.4           | 40.2   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/faster_rcnn_hrnetv2p_w32_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w32_1x_coco/faster_rcnn_hrnetv2p_w32_1x_coco_20200130-6e286425.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w32_1x_coco/faster_rcnn_hrnetv2p_w32_1x_coco_20200130_204442.log.json) |
|   HRNetV2p-W32  | pytorch |   2x    | 9.0        |              | 41.4   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/faster_rcnn_hrnetv2p_w32_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w32_2x_coco/faster_rcnn_hrnetv2p_w32_2x_coco_20200529_015927-976a9c15.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w32_2x_coco/faster_rcnn_hrnetv2p_w32_2x_coco_20200529_015927.log.json)  |
|   HRNetV2p-W40  | pytorch |   1x    | 10.4     | 10.5           | 41.2   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/faster_rcnn_hrnetv2p_w40_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w40_1x_coco/faster_rcnn_hrnetv2p_w40_1x_coco_20200210-95c1f5ce.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w40_1x_coco/faster_rcnn_hrnetv2p_w40_1x_coco_20200210_125315.log.json) |
|   HRNetV2p-W40  | pytorch |   2x    | 10.4     |                |  42.1  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/faster_rcnn_hrnetv2p_w40_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w40_2x_coco/faster_rcnn_hrnetv2p_w40_2x_coco_20200512_161033-0f236ef4.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w40_2x_coco/faster_rcnn_hrnetv2p_w40_2x_coco_20200512_161033.log.json)  |

### Mask R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :-------------:|:------:| :------:|:------:|:--------:|
|   HRNetV2p-W18  | pytorch |   1x    | 7.0      | 11.7           | 37.7   | 34.2    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/mask_rcnn_hrnetv2p_w18_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w18_1x_coco/mask_rcnn_hrnetv2p_w18_1x_coco_20200205-1c3d78ed.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w18_1x_coco/mask_rcnn_hrnetv2p_w18_1x_coco_20200205_232523.log.json) |
|   HRNetV2p-W18  | pytorch |   2x    | 7.0      | -              | 39.8   | 36.0    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/mask_rcnn_hrnetv2p_w18_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w18_2x_coco/mask_rcnn_hrnetv2p_w18_2x_coco_20200212-b3c825b1.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w18_2x_coco/mask_rcnn_hrnetv2p_w18_2x_coco_20200212_134222.log.json) |
|   HRNetV2p-W32  | pytorch |   1x    | 9.4      | 11.3           | 41.2   | 37.1    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/mask_rcnn_hrnetv2p_w32_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w32_1x_coco/mask_rcnn_hrnetv2p_w32_1x_coco_20200207-b29f616e.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w32_1x_coco/mask_rcnn_hrnetv2p_w32_1x_coco_20200207_055017.log.json) |
|   HRNetV2p-W32  | pytorch |   2x    | 9.4      | -              | 42.5   | 37.8    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/mask_rcnn_hrnetv2p_w32_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w32_2x_coco/mask_rcnn_hrnetv2p_w32_2x_coco_20200213-45b75b4d.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w32_2x_coco/mask_rcnn_hrnetv2p_w32_2x_coco_20200213_150518.log.json) |
|   HRNetV2p-W40  | pytorch |   1x    |  10.9    |                | 42.1   |  37.5   |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/mask_rcnn_hrnetv2p_w40_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w40_1x_coco/mask_rcnn_hrnetv2p_w40_1x_coco_20200511_015646-66738b35.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w40_1x_coco/mask_rcnn_hrnetv2p_w40_1x_coco_20200511_015646.log.json)  |
|   HRNetV2p-W40  | pytorch |   2x    |   10.9   |                | 42.8   |  38.2   |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/mask_rcnn_hrnetv2p_w40_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w40_2x_coco/mask_rcnn_hrnetv2p_w40_2x_coco_20200512_163732-aed5e4ab.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w40_2x_coco/mask_rcnn_hrnetv2p_w40_2x_coco_20200512_163732.log.json)  |

### Cascade R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :-------------:|:------:| :------: | :--------: |
|   HRNetV2p-W18  | pytorch |   20e   |  7.0     | 11.0           | 41.2   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/cascade_rcnn_hrnetv2p_w18_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_rcnn_hrnetv2p_w18_20e_coco/cascade_rcnn_hrnetv2p_w18_20e_coco_20200210-434be9d7.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_rcnn_hrnetv2p_w18_20e_coco/cascade_rcnn_hrnetv2p_w18_20e_coco_20200210_105632.log.json)  |
|   HRNetV2p-W32  | pytorch |   20e   |  9.4     | 11.0           | 43.3   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/cascade_rcnn_hrnetv2p_w32_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_rcnn_hrnetv2p_w32_20e_coco/cascade_rcnn_hrnetv2p_w32_20e_coco_20200208-928455a4.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_rcnn_hrnetv2p_w32_20e_coco/cascade_rcnn_hrnetv2p_w32_20e_coco_20200208_160511.log.json)  |
|   HRNetV2p-W40  | pytorch |   20e   |  10.8    |                | 43.8   |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/cascade_rcnn_hrnetv2p_w40_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_rcnn_hrnetv2p_w40_20e_coco/cascade_rcnn_hrnetv2p_w40_20e_coco_20200512_161112-75e47b04.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_rcnn_hrnetv2p_w40_20e_coco/cascade_rcnn_hrnetv2p_w40_20e_coco_20200512_161112.log.json)  |

### Cascade Mask R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :-------------:|:------:| :------:|:------:|:--------:|
|   HRNetV2p-W18  | pytorch |   20e   | 8.5      | 8.5            |41.6    |36.4     |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/cascade_mask_rcnn_hrnetv2p_w18_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_mask_rcnn_hrnetv2p_w18_20e_coco/cascade_mask_rcnn_hrnetv2p_w18_20e_coco_20200210-b543cd2b.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_mask_rcnn_hrnetv2p_w18_20e_coco/cascade_mask_rcnn_hrnetv2p_w18_20e_coco_20200210_093149.log.json)  |
|   HRNetV2p-W32  | pytorch |   20e   |          | 8.3            |44.3    |38.6     |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/cascade_mask_rcnn_hrnetv2p_w32_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_mask_rcnn_hrnetv2p_w32_20e_coco/cascade_mask_rcnn_hrnetv2p_w32_20e_coco_20200512_154043-39d9cf7b.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_mask_rcnn_hrnetv2p_w32_20e_coco/cascade_mask_rcnn_hrnetv2p_w32_20e_coco_20200512_154043.log.json)  |
|   HRNetV2p-W40  | pytorch |   20e   | 12.5     |                |45.1    |39.3     |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/cascade_mask_rcnn_hrnetv2p_w40_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_mask_rcnn_hrnetv2p_w40_20e_coco/cascade_mask_rcnn_hrnetv2p_w40_20e_coco_20200527_204922-969c4610.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_mask_rcnn_hrnetv2p_w40_20e_coco/cascade_mask_rcnn_hrnetv2p_w40_20e_coco_20200527_204922.log.json)    |

### Hybrid Task Cascade (HTC)

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :-------------:|:------:| :------:|:------:|:--------:|
|   HRNetV2p-W18  | pytorch |   20e   | 10.8     | 4.7            | 42.8   | 37.9    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/htc_hrnetv2p_w18_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/htc_hrnetv2p_w18_20e_coco/htc_hrnetv2p_w18_20e_coco_20200210-b266988c.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/htc_hrnetv2p_w18_20e_coco/htc_hrnetv2p_w18_20e_coco_20200210_182735.log.json) |
|   HRNetV2p-W32  | pytorch |   20e   | 13.1     | 4.9            | 45.4   | 39.9    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/htc_hrnetv2p_w32_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/htc_hrnetv2p_w32_20e_coco/htc_hrnetv2p_w32_20e_coco_20200207-7639fa12.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/htc_hrnetv2p_w32_20e_coco/htc_hrnetv2p_w32_20e_coco_20200207_193153.log.json) |
|   HRNetV2p-W40  | pytorch |   20e   | 14.6     |                | 46.4   | 40.8    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/htc_hrnetv2p_w40_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/htc_hrnetv2p_w40_20e_coco/htc_hrnetv2p_w40_20e_coco_20200529_183411-417c4d5b.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/htc_hrnetv2p_w40_20e_coco/htc_hrnetv2p_w40_20e_coco_20200529_183411.log.json) |

### FCOS

| Backbone  | Style   |  GN     | MS train | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:---------:|:-------:|:-------:|:--------:|:-------:|:------:|:------:|:------:|:------:|:--------:|
|HRNetV2p-W18| pytorch | Y       | N       | 1x       | 13.0 | 12.9 | 35.3   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/fcos_hrnetv2p_w18_gn-head_4x4_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w18_gn-head_4x4_1x_coco/fcos_hrnetv2p_w18_gn-head_4x4_1x_coco_20201212_100710-4ad151de.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w18_gn-head_4x4_1x_coco/fcos_hrnetv2p_w18_gn-head_4x4_1x_coco_20201212_100710.log.json) |
|HRNetV2p-W18| pytorch | Y       | N       | 2x       | 13.0 | -    | 38.2   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/fcos_hrnetv2p_w18_gn-head_4x4_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w18_gn-head_4x4_2x_coco/fcos_hrnetv2p_w18_gn-head_4x4_2x_coco_20201212_101110-5c575fa5.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w18_gn-head_4x4_2x_coco/fcos_hrnetv2p_w18_gn-head_4x4_2x_coco_20201212_101110.log.json) |
|HRNetV2p-W32| pytorch | Y       | N       | 1x       | 17.5 | 12.9 | 39.5   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/fcos_hrnetv2p_w32_gn-head_4x4_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w32_gn-head_4x4_1x_coco/fcos_hrnetv2p_w32_gn-head_4x4_1x_coco_20201211_134730-cb8055c0.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w32_gn-head_4x4_1x_coco/fcos_hrnetv2p_w32_gn-head_4x4_1x_coco_20201211_134730.log.json) |
|HRNetV2p-W32| pytorch | Y       | N       | 2x       | 17.5 | -    | 40.8   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/fcos_hrnetv2p_w32_gn-head_4x4_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w32_gn-head_4x4_2x_coco/fcos_hrnetv2p_w32_gn-head_4x4_2x_coco_20201212_112133-77b6b9bb.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w32_gn-head_4x4_2x_coco/fcos_hrnetv2p_w32_gn-head_4x4_2x_coco_20201212_112133.log.json) |
|HRNetV2p-W18| pytorch | Y       | Y       | 2x       | 13.0 | 12.9 | 38.3   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/fcos_hrnetv2p_w18_gn-head_mstrain_640-800_4x4_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w18_gn-head_mstrain_640-800_4x4_2x_coco/fcos_hrnetv2p_w18_gn-head_mstrain_640-800_4x4_2x_coco_20201212_111651-441e9d9f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w18_gn-head_mstrain_640-800_4x4_2x_coco/fcos_hrnetv2p_w18_gn-head_mstrain_640-800_4x4_2x_coco_20201212_111651.log.json) |
|HRNetV2p-W32| pytorch | Y       | Y       | 2x       | 17.5 | 12.4 | 41.9   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/fcos_hrnetv2p_w32_gn-head_mstrain_640-800_4x4_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w32_gn-head_mstrain_640-800_4x4_2x_coco/fcos_hrnetv2p_w32_gn-head_mstrain_640-800_4x4_2x_coco_20201212_090846-b6f2b49f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w32_gn-head_mstrain_640-800_4x4_2x_coco/fcos_hrnetv2p_w32_gn-head_mstrain_640-800_4x4_2x_coco_20201212_090846.log.json) |
|HRNetV2p-W48| pytorch | Y       | Y       | 2x       | 20.3 | 10.8 | 42.7   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco/fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco_20201212_124752-f22d2ce5.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco/fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco_20201212_124752.log.json) |

**Note:**

- The `28e` schedule in HTC indicates decreasing the lr at 24 and 27 epochs, with a total of 28 epochs.
- HRNetV2 ImageNet pretrained models are in [HRNets for Image Classification](https://github.com/HRNet/HRNet-Image-Classification).
