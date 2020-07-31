_base_ = ['./yolact_r50.py']

model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))

work_dir = './work_dirs/yolact_r101'
