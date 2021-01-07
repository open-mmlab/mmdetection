_base_ = ['../mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py', '../_base_/swa.py']
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))

only_swa_training = True
swa_training = True
swa_load_from = 'checkpoints/mask_rcnn/mask_rcnn_r101_fpn_2x_coco_' \
                'bbox_mAP-0.408__segm_mAP-0.366_20200505_071027-14b391c7.pth'
swa_resume_from = None

# swa optimizer
swa_optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# swa learning policy
swa_lr_config = dict(
    policy='cyclic',
    target_ratio=(1, 0.01),
    cyclic_times=24,
    step_ratio_up=0.0)
swa_total_epochs = 24
