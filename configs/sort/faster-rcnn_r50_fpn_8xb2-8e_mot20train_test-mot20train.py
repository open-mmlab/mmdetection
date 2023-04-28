_base_ = ['./faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval']
model = dict(
    rpn_head=dict(bbox_coder=dict(clip_border=True)),
    roi_head=dict(
        bbox_head=dict(bbox_coder=dict(clip_border=True), num_classes=1)))
# data
data_root = 'data/MOT20/'
train_dataloader = dict(
    dataset=dict(
        data_root=data_root, ann_file='annotations/train_cocoformat.json'))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root, ann_file='annotations/train_cocoformat.json'))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/train_cocoformat.json')
test_evaluator = val_evaluator

# training schedule for 8e
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=8, val_interval=1)

# learning rate
param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=100),
    dict(
        type='MultiStepLR',
        begin=0,
        end=8,
        by_epoch=True,
        milestones=[6],
        gamma=0.1)
]
