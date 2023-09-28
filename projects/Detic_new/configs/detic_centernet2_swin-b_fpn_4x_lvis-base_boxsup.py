_base_ = './detic_centernet2_swin-b_fpn_4x_lvis_boxsup.py'

# 'lvis_v1_train_norare.json' is the annotations of lvis_v1
# removing the labels of 337 rare-class
train_dataloader = dict(
    dataset=dict(
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(ann_file='annotations/lvis_v1_train_norare.json')))
