_base_ = ['faster-rcnn_r50_fpn_32xb2-1x_openimages.py']

model = dict(
    roi_head=dict(bbox_head=dict(num_classes=500)),
    test_cfg=dict(rcnn=dict(score_thr=0.01)))

# dataset settings
dataset_type = 'OpenImagesChallengeDataset'
train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file='challenge2019/challenge-2019-train-detection-bbox.txt',
        label_file='challenge2019/cls-label-description.csv',
        hierarchy_file='challenge2019/class_label_tree.np',
        meta_file='challenge2019/challenge-2019-train-metas.pkl'))
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file='challenge2019/challenge-2019-validation-detection-bbox.txt',
        data_prefix=dict(img='OpenImages/'),
        label_file='challenge2019/cls-label-description.csv',
        hierarchy_file='challenge2019/class_label_tree.np',
        meta_file='challenge2019/challenge-2019-validation-metas.pkl',
        image_level_ann_file='challenge2019/challenge-2019-validation-'
        'detection-human-imagelabels.csv'))
test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file='challenge2019/challenge-2019-validation-detection-bbox.txt',
        label_file='challenge2019/cls-label-description.csv',
        hierarchy_file='challenge2019/class_label_tree.np',
        meta_file='challenge2019/challenge-2019-validation-metas.pkl',
        image_level_ann_file='challenge2019/challenge-2019-validation-'
        'detection-human-imagelabels.csv'))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (32 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)
