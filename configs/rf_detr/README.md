# RF-DETR in MMDetection

This directory adds RF-DETR Small as a native MMDetection detector.

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
