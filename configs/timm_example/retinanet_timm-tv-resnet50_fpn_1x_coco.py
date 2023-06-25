_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# please install mmpretrain
# import mmpretrain.models to trigger register_module in mmpretrain
custom_imports = dict(
    imports=['mmpretrain.models'], allow_failed_imports=False)

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmpretrain.TIMMBackbone',
        model_name='tv_resnet50',  # ResNet-50 with torchvision weights
        features_only=True,
        pretrained=True,
        out_indices=(1, 2, 3, 4)))

# optimizer
optim_wrapper = dict(optimizer=dict(lr=0.01))
