_base_ = '../mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
# fp16 settings
fp16 = dict(loss_scale=512.)
work_dir = './work_dirs/mask_rcnn_r50_fpn_fp16_1x'
