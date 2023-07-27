_base_ = ['./oneformer_swin-l-p4-w12-384-in21k-lsj_panoptic.py']

model = dict(panoptic_head=dict(task='semantic'), )
