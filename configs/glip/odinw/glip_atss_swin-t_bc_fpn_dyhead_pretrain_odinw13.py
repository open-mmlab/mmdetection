_base_ = './glip_atss_swin-t_a_fpn_dyhead_pretrain_odinw13.py'

model = dict(bbox_head=dict(early_fuse=True))
