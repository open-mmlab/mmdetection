# RF-Next: Efficient Receptive Field Search for CNN

> [RF-Next: Efficient Receptive Field Search for Convolutional Neural Networks](http://mftp.mmcheng.net/Papers/22TPAMI-ActionSeg.pdf)

## Abstract

Temporal/spatial receptive fields of models play an important role in sequential/spatial tasks. Large receptive fields facilitate long-term relations, while small receptive fields help to capture the local details. Existing methods construct models with hand-designed receptive fields in layers. Can we effectively search for receptive field combinations to replace hand-designed patterns? To answer this question, we propose to find better receptive field combinations through a global-to-local search scheme. Our search scheme exploits both global search to find the coarse combinations and local search to get the refined receptive field combinations further. The global search finds possible coarse combinations other than human-designed patterns. On top of the global search, we propose an expectation-guided iterative local search scheme to refine combinations effectively. Our RF-Next models, plugging receptive field search to various models, boost the performance on many tasks, e.g., temporal action segmentation, object detection, instance segmentation, and speech synthesis.
The source code is publicly available on [http://mmcheng.net/rfnext](http://mmcheng.net/rfnext).

## Results and Models

### ConvNext on COCO

|   Backbone    |       Method       |     RFNext      | Lr Schd | box mAP | mask mAP |                                                                                                                                                                                                          Config                                                                                                                                                                                                           |                                                                              Download                                                                              |
| :-----------: | :----------------: | :-------------: | :-----: | :-----: | :------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  ConvNeXt-T   | Cascade Mask R-CNN |       NO        |   3x   |  50.4   |   43.7   |                                                                                                                                                                     [config](https://github.com/facebookresearch/ConvNeXt/tree/main/object_detection)                                                                                                                                                                     | [model](https://github.com/facebookresearch/ConvNeXt/tree/main/object_detection) \| [log](https://github.com/facebookresearch/ConvNeXt/tree/main/object_detection) |
| RF-ConvNeXt-T | Cascade Mask R-CNN |  Single-Branch  |   3x   |  50.6   |   44.0   | [search](https://github.com/ShangHua-Gao/RF-mmdetection/tree/rfsearch/configs/rfnext/rfnext_search_cascade_mask_rcnn_convnext_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in1k.py) [retrain](https://github.com/ShangHua-Gao/RF-mmdetection/tree/rfsearch/configs/rfnext/rfnext_fixed_single_branch_cascade_mask_rcnn_convnext_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in1k.py) |                                                                      [model](<>) \| [log](<>)                                                                      |
| RF-ConvNeXt-T | Cascade Mask R-CNN | Multiple-Branch |   3x   |  50.9   |   44.3   | [search](https://github.com/ShangHua-Gao/RF-mmdetection/tree/rfsearch/configs/rfnext/rfnext_search_cascade_mask_rcnn_convnext_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in1k.py) [retrain](https://github.com/ShangHua-Gao/RF-mmdetection/tree/rfsearch/configs/rfnext/rfnext_fixed_multi_branch_cascade_mask_rcnn_convnext_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in1k.py)  |                                                                      [model](<>) \| [log](<>)                                                                      |

### PVTv2 on COCO

|  Backbone   |   Method   |     RFNext      | Lr Schd | box mAP | mask mAP |                                                                                                                                          Config                                                                                                                                           |                                                       Download                                                       |
| :---------: | :--------: | :-------------: | :-----: | :-----: | :------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------: |
|  PVTv2-b0   | Mask R-CNN |       NO        |   1x    |  38.2   |   36.2   |                                                                                                                [config](https://github.com/whai362/PVT/tree/v2/detection)                                                                                                                 | [model](https://github.com/whai362/PVT/tree/v2/detection) \| [log](https://github.com/whai362/PVT/tree/v2/detection) |
| RF-PVTv2-b0 | Mask R-CNN |  Single-Branch  |   1x    |  38.8   |   36.8   | [search](https://github.com/ShangHua-Gao/RF-mmdetection/tree/rfsearch/configs/rfnext/rfnext_search_mask_rcnn_pvtv2-b0_fpn_1x_coco.py) [retrain](https://github.com/ShangHua-Gao/RF-mmdetection/tree/rfsearch/configs/rfnext/rfnext_fixed_single_branch_mask_rcnn_pvtv2-b0_fpn_1x_coco.py) |                                               [model](<>) \| [log](<>)                                               |
| RF-PVTv2-b0 | Mask R-CNN | Multiple-Branch |   1x    |  39.1   |   37.1   | [search](https://github.com/ShangHua-Gao/RF-mmdetection/tree/rfsearch/configs/rfnext/rfnext_search_mask_rcnn_pvtv2-b0_fpn_1x_coco.py) [retrain](https://github.com/ShangHua-Gao/RF-mmdetection/tree/rfsearch/configs/rfnext/rfnext_fixed_multi_branch_mask_rcnn_pvtv2-b0_fpn_1x_coco.py)  |                                               [model](<>) \| [log](<>)                                               |

### Res2Net on COCO

|    Backbone    |       Method       |     RFNext      | Lr Schd | box mAP | mask mAP |                                                                                                                                                  Config                                                                                                                                                  |         Download         |
| :------------: | :----------------: | :-------------: | :-----: | :-----: | :------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------: |
|  Res2Net-101   | Cascade Mask R-CNN |       NO        |   20e   |  46.3   |   40.0   |                                                                                     [config](https://github.com/ShangHua-Gao/RF-mmdetection/tree/rfsearch/configs/res2net/cascade_mask_rcnn_r2_101_fpn_20e_coco.py)                                                                                      | [model](<>) \| [log](<>) |
| RF-Res2Net-101 | Cascade Mask R-CNN |  Single-Branch  |   20e   |  46.9   |   40.7   | [search](https://github.com/ShangHua-Gao/RF-mmdetection/tree/rfsearch/configs/rfnext/rfnext_search_cascade_mask_rcnn_r2_101_fpn_20e_coco.py)  [retrain](https://github.com/ShangHua-Gao/RF-mmdetection/tree/rfsearch/configs/rfnext/rfnext_fixed_single_branch_cascade_mask_rcnn_r2_101_fpn_20e_coco.py) | [model](<>) \| [log](<>) |
| RF-Res2Net-101 | Cascade Mask R-CNN | Multiple-Branch |   20e   |  47.9   |   41.5   | [search](https://github.com/ShangHua-Gao/RF-mmdetection/tree/rfsearch/configs/rfnext/rfnext_search_cascade_mask_rcnn_r2_101_fpn_20e_coco.py)  [retrain](https://github.com/ShangHua-Gao/RF-mmdetection/tree/rfsearch/configs/rfnext/rfnext_fixed_multi_branch_cascade_mask_rcnn_r2_101_fpn_20e_coco.py)  | [model](<>) \| [log](<>) |

### HRNet on COCO

|    Backbone     |       Method       |     RFNext      | Lr Schd | box mAP | mask mAP |                                                                                                                                                       Config                                                                                                                                                        |         Download         |
| :-------------: | :----------------: | :-------------: | :-----: | :-----: | :------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------: |
|  HRNetV2p-W18   | Cascade Mask R-CNN |       NO        |   20e   |  41.6   |   36.4   |                                                                                           [config](https://github.com/ShangHua-Gao/RF-mmdetection/tree/rfsearch/configs/hrnet/cascade_mask_rcnn_hrnetv2p_w18_20e_coco.py)                                                                                           | [model](<>) \| [log](<>) |
| RF-HRNetV2p-W18 | Cascade Mask R-CNN |  Single-Branch  |   20e   |  42.9   |   37.6   | [search](https://github.com/ShangHua-Gao/RF-mmdetection/tree/rfsearch/configs/rfnext/rfsearched_search_cascade_mask_rcnn_hrnetv2p_w18_20e_coco.py) [retrain](https://github.com/ShangHua-Gao/RF-mmdetection/tree/rfsearch/configs/rfnext/rfsearched_fixed_single_branch_cascade_mask_rcnn_hrnetv2p_w18_20e_coco.py) | [model](<>) \| [log](<>) |
| RF-HRNetV2p-W18 | Cascade Mask R-CNN | Multiple-Branch |   20e   |  43.7   |   38.1   | [search](https://github.com/ShangHua-Gao/RF-mmdetection/tree/rfsearch/configs/rfnext/rfsearched_search_cascade_mask_rcnn_hrnetv2p_w18_20e_coco.py) [retrain](https://github.com/ShangHua-Gao/RF-mmdetection/tree/rfsearch/configs/rfnext/rfsearched_fixed_multi_branch_cascade_mask_rcnn_hrnetv2p_w18_20e_coco.py)  | [model](<>) \| [log](<>) |

Note: the performance of multi-branch models listed above are evaluated during searching to save computional cost, retraining would achieve similar or better performance.

### Res2Net on COCO panoptic

| Backbone | Method | RFNext | Lr schd |  PQ  |  SQ  |  RQ  |                                                            Config                                                             |                                                                                                                                                                          Download                                                                                                                                                                          |
| :-------: | :-----: | :-----: | :-----: | :--: | :--: | :--: | :---------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Res2Net-50 | Panoptic FPN |  NO  |  1x  | 42.5 | 78.0 | 51.8 |     [config](https://github.com/ShangHua-Gao/RF-mmdetection/tree/rfsearch/configs/panoptic_fpn/panoptic_fpn_r2_50_fpn_fp16_1x_coco.py)      |                   [model]() \| [log]()                   |
| RF-Res2Net-50 | Panoptic FPN |  Single-Branch  |  1x  | 44.0 | 78.7 | 53.6 |     [search](https://github.com/ShangHua-Gao/RF-mmdetection/tree/rfsearch/configs/rfnext/rfnext_search_panoptic_fpn_r2_50_fpn_fp16_1x_coco.py) [retrain](https://github.com/ShangHua-Gao/RF-mmdetection/tree/rfsearch/configs/rfnext/rfnext_fixed_single_branch_panoptic_fpn_r2_50_fpn_fp16_1x_coco.py)      |                   [model]() \| [log]()                   |
| RF-Res2Net-50 | Panoptic FPN |  Multiple-Branch  |  1x  | 44.4 | 79.0 | 54.0 |     [search](https://github.com/ShangHua-Gao/RF-mmdetection/tree/rfsearch/configs/rfnext/rfnext_search_panoptic_fpn_r2_50_fpn_fp16_1x_coco.py) [retrain](https://github.com/ShangHua-Gao/RF-mmdetection/tree/rfsearch/configs/rfnext/rfnext_fixed_multi_branch_panoptic_fpn_r2_50_fpn_fp16_1x_coco.py)      |                   [model]() \| [log]()                   |

## Configs

If you want to search receptive fields on an existing model, you need to define `rfsearch_cfg` in the `model` of config file and then define a hook `RFSearch`.

```python
rfsearch_cfg=dict(
    logdir='the path to save searched structures',
    mode='search',
    rfstructure_file=None,
    config=dict(
        search=dict(
            step=0,
            max_step=11,
            search_interval=1,
            exp_rate=0.5,
            init_alphas=0.01,
            mmin=1,
            mmax=24,
            num_branches=2,
            skip_layer=[])),
    )

custom_hooks = [
    dict(
        type='RFSearch',
        logdir=model['rfsearch_cfg']['logdir'],
        config=model['rfsearch_cfg']['config'],
        mode=model['rfsearch_cfg']['mode'],
    ),
]
```

Arguments:

- `max_step`: The maximum number of steps to update the structures.
- `search_interval`: The interval (epoch) between two updates.
- `exp_rate`:  The controller of the sparsity of search space. For a conv with an initial dilation rate of `D`, dilation rates will be sampled with an interval of `exp_rate * D`.
- `num_branches`: The controller of the size of search space (the number of branches). If you set `S=3`, the dilations are `[D - exp_rate * D, D, D + exp_rate * D]` for three branches. If you set `num_branches=2`, the dilations are `[D - exp_rate * D, D + exp_rate * D]`. With `num_branches=2`, you can achieve similar performance with less MEMORY and FLOPS.
- `skip_layer`: The modules in skip_layer will be ignored during the receptive field search.

## Training

### Searching Jobs

You can launch searching jobs by using config files with prefix `rfnext_search`. The json files of searched structures will be saved to `rfsearch_cfg.log_dir`.

If you want to further search receptive fields upon a searched structure, please set `rfsearch_cfg.rfstructure_file` in config file to the corresponding json file.

```shell
bash ./tools/dist_train.sh \
    ${rfnext_search_CONFIG_FILE} \
    ${GPU_NUM}
```

### Training Jobs

Setting `rfsearch_cfg.rfstructure_file` to the searched structure file (.json) and setting `rfsearch_cfg.mode` to `fixed_single_branch` or `fixed_multi_branch`, you can retrain a model with the searched structure.
You can launch fixed_single_branch/fixed_multi_branch training jobs by using config files with prefix `rfnext_fixed_single_branch` or `rfnext_fixed_multi_branch`.

```shell
bash ./tools/dist_train.sh \
    ${rfnext_fixed_single_branch_CONFIG_FILE} \
    ${GPU_NUM}

bash ./tools/dist_train.sh \
    ${rfnext_fixed_multi_branch_CONFIG_FILE} \
    ${GPU_NUM}
```

Note that the models after the searching stage is ready a `fixed_multi_branch` version, which achieves better performance than `fixed_single_branch`, without any retraining.

**Warning: Please do not remove the `rfsearch_cfg` in the `model` of config file, because we are using the hook that only apply the settings in `rfsearch_cfg`.**

## Inference

`rfsearch_cfg.rfstructure_file` and `rfsearch_cfg.mode` should be set for inferencing stage.

Single branch inference:

```shell
./tools/dist_test.sh \
    ${rfnext_fixed_single_branch_CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${GPU_NUM} \
    --out results.pkl \
    --eval bbox segm \
    --options "classwise=True"

```

If you want to inference models with multiple-branch structure, please set `rfsearch_cfg.mode` to `fixed_multi_branch`.

```shell
./tools/dist_test.sh \
    ${rfnext_fixed_multi_branch_CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${GPU_NUM} \
    --out results.pkl \
    --eval bbox segm \
    --options "classwise=True"

```

## Citation

```
@article{gao2022rfnext,
title={RF-Next: Efficient Receptive Field Search for Convolutional Neural Networks},
author={Gao, Shanghua and Li, Zhong-Yu and Han, Qi and Cheng, Ming-Ming and Wang, Liang},
journal=TPAMI,
year={2022}
}

@inproceedings{gao2021global2local,
  title     = {Global2Local: Efficient Structure Search for Video Action Segmentation},
  author    = {Gao, Shanghua and Han, Qi and Li, Zhong-Yu and Peng, Pai and Wang, Liang and Cheng, Ming-Ming},
  booktitle = CVPR,
  year      = {2021}
}
```
