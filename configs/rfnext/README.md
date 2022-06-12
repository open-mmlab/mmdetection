# RF-Next: Efficient Receptive Field Search for CNN

> [RF-Next: Efficient Receptive Field Search for Convolutional Neural Networks]()

## Abstract

Temporal/spatial receptive fields of models play an important role in sequential/spatial tasks. Large receptive fields facilitate long-term relations, while small receptive fields help to capture the local details. Existing methods construct models with hand-designed receptive fields in layers. Can we effectively search for receptive field combinations to replace hand-designed patterns? To answer this question, we propose to find better receptive field combinations through a global-to-local search scheme. Our search scheme exploits both global search to find the coarse combinations and local search to get the refined receptive field combinations further. The global search finds possible coarse combinations other than human-designed patterns. On top of the global search, we propose an expectation-guided iterative local search scheme to refine combinations effectively. Our RF-Next models, plugging receptive field search to various models, boost the performance on many tasks, e.g., temporal action segmentation, object detection, instance segmentation, and speech synthesis.
The source code is publicly available on [http://mmcheng.net/rfnext](http://mmcheng.net/rfnext).

## Results and Models

### ConvNext on COCO

| Backbone | Method | RFNext | Lr Schd | box mAP | mask mAP | Config | Download |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|:---:|
| ConvNeXt-T | Cascade Mask R-CNN | NO            | 20e | 50.4 | 43.7 | [config](configs/convnext/cascade_mask_rcnn_convnext_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in1k.py) | [model]() &#124; [log]() |
| RF-ConvNeXt-T | Cascade Mask R-CNN | Single-Branch | 20e | 50.6 | 44.0 | [config](configs/convnext/rfsearch_cascade_mask_rcnn_convnext_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in1k.py) | [model]() &#124; [log]() |
| RF-ConvNeXt-T | Cascade Mask R-CNN | Multiple-Branch  | 20e | 50.9 | 44.3 | [config](configs/convnext/rfsearch_cascade_mask_rcnn_convnext_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in1k.py) | [model]() &#124; [log]() |

### PVTv2 on COCO

| Backbone | Method | RFNext | Lr Schd | box mAP | mask mAP | Config | Download |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|:---:|
| PVTv2-b0 | Mask R-CNN	 | NO            | 1x | 38.2 | 36.2 | [config]() | [model]() &#124; [log]() |
| RF-PVTv2-b0| Mask R-CNN	 | Single-Branch | 1x | 38.8 | 36.8 | [config](configs/pvt/rfsearch_mask_rcnn_pvtv2-b0_fpn_1x_coco.py) | [model]() &#124; [log]() |
| RF-PVTv2-b0 | Mask R-CNN	 | Multiple-Branch  | 1x | 39.1 | 37.1 | [config](configs/pvt/rfsearch_mask_rcnn_pvtv2-b0_fpn_1x_coco.py) | [model]() &#124; [log]() |

### Res2Net on COCO

| Backbone | Method | RFNext | Lr Schd | box mAP | mask mAP | Config | Download |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|:---:|
| Res2Net-101 | Cascade Mask R-CNN	 | NO            | 20e | 46.3 | 40.0 | [config](configs/res2net/cascade_mask_rcnn_r2_101_fpn_20e_coco.py) | [model]() &#124; [log]() |
| RF-Res2Net-101 | Cascade Mask R-CNN	 | Single-Branch | 20e | 46.9 | 40.7 | [config](configs/res2net/rfsearched_cascade_mask_rcnn_r2_101_fpn_20e_coco.py) | [model]() &#124; [log]() |
| RF-Res2Net-101 | Cascade Mask R-CNN	 | Multiple-Branch  | 20e | 47.9 | 41.5 | [config](configs/res2net/rfsearched_cascade_mask_rcnn_r2_101_fpn_20e_coco.py) | [model]() &#124; [log]() |

### HRNet on COCO

| Backbone | Method | RFNext | Lr Schd | box mAP | mask mAP | Config | Download |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|:---:|
| HRNetV2p-W18 | Cascade Mask R-CNN	 | NO            | 20e | 41.6 | 36.4 | [config](configs/hrnet/cascade_mask_rcnn_hrnetv2p_w18_20e_coco.py) | [model]() &#124; [log]() |
| RF-HRNetV2p-W18 | Cascade Mask R-CNN	 | Single-Branch | 20e | 42.9 | 37.6 | [config](configs/hrnet/rfsearched_cascade_mask_rcnn_hrnetv2p_w18_20e_coco.py) | [model]() &#124; [log]() |
| RF-HRNetV2p-W18 | Cascade Mask R-CNN	 | Multiple-Branch  | 20e | 43.7 | 38.1 | [config](configs/hrnet/rfsearched_cascade_mask_rcnn_hrnetv2p_w18_20e_coco.py) | [model]() &#124; [log]() |

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
            normlize='absavg',
            mmin=1,
            mmax=24,
            S=2,
            finetune=False,
            skip_layer=[])),
    )

custom_imports = dict(
    imports=['mmcv.cnn.rfsearch'], allow_failed_imports=False)
custom_hooks = [
    dict(type='NumClassCheckHook'),
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
- `S`: The controller of the size of search space (the number of branches). If you set `S=3`, the dilations are `[D - exp_rate * D, D, D + exp_rate * D]` for three branches. If you set `S=2`, the dilations are `[D - exp_rate * D, D + exp_rate * D]`. With `S=2`, you can achieve similar performance with less MEMORY and FLOPS.
- `skip_layer`: The modules in skip_layer will be ignored during the receptive field search.

## Training
### Searching Jobs
You can launch searching jobs by using config files with `rfsearch_cfg`. The json files of searched structures will be saved to `rfsearch_cfg.log_dir`.

If you want to further search receptive fields upon a searched structure, please set `rfsearch_cfg.rfstructure_file` to the corresponding json file.

### Finetuning Jobs

Setting `rfsearch_cfg.rfstructure_file` to the searched structure file (.json) and setting `rfsearch_cfg.mode` to `fixed_single_branch`, you can retrain a model with the searched structure.

```shell
bash ./tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    --cfg-options model.rfsearch_cfg.rfstructure_file="the json file" model.rfsearch_cfg.mode="fixed_single_branch" \
    --work_dir [./work_dirs/rfsearched_*]
```

If you want to retrain a model that keeps the multiple-branch structure as the search stage, you can set `rfsearch_cfg.mode` to `fixed_multi_branch`.

Note that just the models after the search stage could achieve better performance than `fixed_single_branch`, without any retraining.

## Inference
`rfsearch_cfg.rfstructure_file` and `rfsearch_cfg.mode` should be set for inferencing stage.

```shell
./tools/dist_test.sh \
    configs/convnext/rfsearch_cascade_mask_rcnn_convnext_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in1k.py \
    checkpoints/*.pth \
    8 \
    --out results.pkl \
    --eval bbox segm \
    --options "classwise=True" \
    --cfg-options model.rfsearch_cfg.rfstructure_file="the json file" model.rfsearch_cfg.mode="fixed_single_branch"

```

If you want to inference models with multiple-branch structure, please set `rfsearch_cfg.mode` to `fixed_multi_branch`.

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
