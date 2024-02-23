_base_ = './baron_kd_mask_rcnn_r50_fpn_syncbn_45kx4_lvis.py'
kd = dict(
    type='BaronShared4Conv1FCBBoxHead',
    num_words=4,
    word_dim=512,
    in_channels=256,
    fc_out_channels=1024,
    roi_feat_size=7,
    with_reg=False,
    norm_cfg=dict(type='SyncBN', requires_grad=True),
    norm_eval=False,
    init_cfg=dict(
        type='Pretrained',
        prefix='roi_head.bbox_head',
        checkpoint='checkpoints/res50_fpn_soco_star_400.pth'))
model = dict(
    roi_head=dict(
        bbox_head=dict(
            kd=kd,
            type='EnsembleBaronShared4Conv1FCBBoxHead',
            ensemble_factor=2.0 / 3.0,
            class_info='projects/Ovdet/metadata/' +
            'lvis_v1_train_cat_norare_info.json',
            transfer_factor=None), ), )
