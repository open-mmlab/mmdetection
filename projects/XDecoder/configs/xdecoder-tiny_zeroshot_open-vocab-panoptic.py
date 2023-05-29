_base_ = 'xdecoder-tiny_zeroshot_open-vocab-semseg.py'

# TODO: Open sets are not easy to handle
model = dict(task='panoptic',
             test_cfg=dict(
                 nms_pre=1000,
                 min_bbox_size=0,
                 score_thr=0.05,
                 nms=dict(type='nms', iou_threshold=0.5),
                 max_per_img=100)
             )
