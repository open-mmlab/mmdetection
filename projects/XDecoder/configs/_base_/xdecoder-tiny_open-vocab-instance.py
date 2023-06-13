_base_ = 'xdecoder-tiny_open-vocab-semseg.py'

model = dict(head=dict(task='instance'), test_cfg=dict(max_per_img=100))
