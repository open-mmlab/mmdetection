_base_ = 'xdecoder-tiny_zeroshot_open-vocab-semseg.py'

model = dict(task='instance',
             test_cfg=dict(
                 nms_pre=1000,
                 min_bbox_size=0,
                 score_thr=0.05,
                 nms=dict(type='nms', iou_threshold=0.5),
                 max_per_img=100)
             )

_base_.test_pipeline[1] = dict(scale=800)