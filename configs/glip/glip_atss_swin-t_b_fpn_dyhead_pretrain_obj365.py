_base_ = './glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365.py'

model = dict(bbox_head=dict(early_fuse=True))
