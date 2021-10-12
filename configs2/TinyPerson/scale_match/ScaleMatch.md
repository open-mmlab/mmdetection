# Scale Match

## config file
### 1. Scale Match for Adap RetinaNet

```shell script
cp configs/retinanet/retinanet_r50_fpn_1x_coco.py configs2/TinyPerson/scale_match/retinanet_r50_fpns4_1x_coco_sm_tinyperson.py
# modified retinanet_r50_fpn_1x_coco_sm_tinyperson.py
1. _base_: '../_base_/datasets/coco_detection.py', correct other relative path.
2. copy content in configs/_base_/datasets/coco_detection.py to retinanet_r50_fpns4_1x_coco_sm_tinyperson.py.py
3. dataset adn transform:
  3.1 dataset_type='CocoDataset'  # for evaluation
  3.2 train pipeline: Resize => ScaleMatchResize
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='ScaleMatchResize', scale_match_type="ScaleMatch",
         anno_file="data/tiny_set/mini_annotations/tiny_set_train_all_erase.json",
         bins=100, default_scale=0.25, scale_range=(0.1, 1)),
  3.3 test_pipeline:  img_scale=(1333, 800) => img_scale=(333, 200)
  3.4 data.train.ann_file => data_root + 'annotations/instances_merge2017.json',
4. adaptive retinanet and anchor:
    model = dict(
    neck=dict(start_level=0, num_outs=5), # start_level=1, 
    bbox_head=dict(
        type='RetinaHead',
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=2,    # 4
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]  # [8, 16, 32, 64, 128]
        )
    )
) 
```

### 2. Monotonicity Scale Match for scale Match  
```shell script
torch configs2/TinyPerson/scale_match/retinanet_r50_fpns4_1x_coco_msm_tinyperson.py

1. _base_ = ['./retinanet_r50_fpns4_1x_coco_sm_tinyperson.py']
2. train_pipeline: ScaleMatchResize => MonotonicityScaleMatch
    ...
    dict(type='ScaleMatchResize', scale_match_type="MonotonicityScaleMatch",
         src_anno_file="data/coco/annotations/instances_train2017.json",
         dst_anno_file="data/tiny_set/mini_annotations/tiny_set_train_all_erase.json",
         bins=100, default_scale=0.25, scale_range=(0.1, 1)),
    ...
  data = dict(train=dict(pipeline=train_pipeline))
```

## performance

- GPU: 3080 x 2
- Adap RetainaNet-c means use clip grad while training.
- COCO val $mmap$ only use for debug, cause val also add to train while sm/msm coco to TinyPerson

detector | type | $AP_{50}^{tiny}$| script | COCO200 val $mmap$ | coco batch/lr
--- | --- | ---| ---| ---| ---
Faster-FPN | - |  ~~47.90~~<br/>49.81 | configs2/TinyPerson/base/Baseline_TinyPerson.sh:exp1.2 | - | -
Faster-FPN | SM | ~~50.06~~<br/>50.85 | ScaleMatch_TinyPerson.sh:exp4.0 | 18.9 | 8x2/0.01
Faster-FPN | SM | ~~49.53~~<br/>50.30 | ScaleMatch_TinyPerson.sh:exp4.1 | 18.5 | 4x2/0.01
Faster-FPN | MSM | ~~49.39~~<br/>50.18 | ScaleMatch_TinyPerson.sh:exp4.2 | 12.1 | 4x2/0.01
--| --| --
Adap RetainaNet-c | -   | ~~43.66~~<br/>45.22 | configs2/TinyPerson/base/Baseline_TinyPerson.sh:exp2.3 | - | -
Adap RetainaNet-c | SM  | ~~50.07~~<br/>51.78 | ScaleMatch_TinyPerson.sh:exp5.1 | 19.6 | 4x2/0.01
Adap RetainaNet-c | MSM | ~~48.39~~<br/>50.00 | ScaleMatch_TinyPerson.sh:exp5.2 | 12.9 | 4x2/0.01

