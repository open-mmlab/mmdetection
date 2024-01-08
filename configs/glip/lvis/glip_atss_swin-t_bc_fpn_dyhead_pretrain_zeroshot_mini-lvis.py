_base_ = './glip_atss_swin-t_a_fpn_dyhead_pretrain_zeroshot_mini-lvis.py'

model = dict(bbox_head=dict(early_fuse=True))
