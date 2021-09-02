_base_ = './mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py'

model = dict(
    backbone=dict(
        drop_path_rate=0.1,
    ),
)

lr_config = dict(step=[8, 11])
runner = dict(max_epochs=12)

