_base_ = './coco_sparse_rcnn_r50_fpn_mstrain_480-800_3x_fskd.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))


# Distillation Params
teacher_config_path = 'result/coco/sparse_rcnn_r101_fpn_mstrain_480-800_3x/coco_sparse_rcnn_r101_fpn_mstrain_480-800_3x.py'
teacher_weight_path = 'result/coco/sparse_rcnn_r101_fpn_mstrain_480-800_3x/epoch_36.pth'
backbone_pretrain = False

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)