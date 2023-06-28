_base_ = 'xdecoder-tiny_open-vocab-semseg.py'

model = dict(
    head=dict(task='panoptic'), test_cfg=dict(mask_thr=0.8, overlap_thr=0.8))
