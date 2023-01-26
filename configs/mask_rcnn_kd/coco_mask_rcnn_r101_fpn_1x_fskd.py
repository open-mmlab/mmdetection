_base_ = './coco_mask_rcnn_r50_fpn_1x_fskd.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
                      

# Distillation Params
teacher_config_path = 'result/coco/mask_rcnn_r101_fpn_1x/coco_mask_rcnn_r101_fpn_1x.py'
teacher_weight_path = 'result/coco/mask_rcnn_r101_fpn_1x/epoch_12.pth'
backbone_pretrain = False


