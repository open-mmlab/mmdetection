_base_ = ['../../../configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py']

custom_imports = dict(imports=['projects.example_project.dummy'])

_base_.model.backbone.type = 'DummyResNet'
