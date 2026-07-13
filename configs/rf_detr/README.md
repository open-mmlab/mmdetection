# RF-DETR in MMDetection

This directory adds RF-DETR as native MMDetection detector configs.

## Train

```bash
python tools/train.py configs/rf_detr/rf_detr_small.py
```

## Test

```bash
python tools/test.py configs/rf_detr/rf_detr_small.py work_dirs/rf_detr_small/latest.pth
```

## Smoke Test

```bash
python tools/train.py configs/rf_detr/rf_detr_small_smoke.py
```

## Configs

Detection:

- `rf_detr_nano.py`
- `rf_detr_small.py`
- `rf_detr_base.py`
- `rf_detr_medium.py`
- `rf_detr_large.py`

Instance segmentation architecture configs:

- `rf_detr_seg_preview.py`
- `rf_detr_seg_nano.py`
- `rf_detr_seg_small.py`
- `rf_detr_seg_medium.py`
- `rf_detr_seg_large.py`
- `rf_detr_seg_xlarge.py`
- `rf_detr_seg_2xlarge.py`

Keypoint architecture configs:

- `rf_detr_keypoint_preview.py`

Each variant selects the matching upstream RF-DETR model config through
`model.model_config_type`.  Detection configs use MMDetection bbox datasets and
COCO bbox evaluation. Segmentation and keypoint configs select RF-DETR's
segmentation/keypoint heads plus matching MMDetection annotation pipelines and
evaluators; the current detector wrapper still returns detection
`pred_instances` for visualization and standard bbox inference.

## Demo Inference

The current converted RF-DETR Small checkpoint was run through MMDetection on
`../test_images/mmdet_demo.jpg`. Outputs were written to:

- `../outdir/rf_detr_small_mmdet_demo.jpg`
- `../outdir/rf_detr_small_mmdet_demo_predictions.npz`

## Convert Official Checkpoint

```bash
python tools/model_converters/rf_detr_to_mmdet.py \
  ../checkpoints/rf-detr-small.pth \
  ../checkpoints/rf-detr-small-mmdet.pth \
  --config configs/rf_detr/rf_detr_small.py \
  --num-classes 90 \
  --allow-missing-prefix model._kp_active_mask
```

The converter writes a mapping report and fails on unexpected important missing
weights or shape mismatches.

## Important Variables

Edit these at the top of `rf_detr_small.py`:

- `data_root`
- `train_ann_file`
- `val_ann_file`
- `test_ann_file`
- `num_classes`
- `metainfo`
- `load_from`

Official RF-DETR Small checkpoints use `num_classes=90` because the upstream
model preserves COCO category-id slots plus a background output.
