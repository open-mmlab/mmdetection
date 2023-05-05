_base_ = './glip_atss_swin-t_fpn_dyhead_pretrain_obj365.py'

model = dict(
    train_cfg=dict(
        early_fuse=True)
)