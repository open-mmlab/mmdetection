_base_ = './scnet_x101_64x4d_fpn_20e_coco.py'
data = dict(samples_per_gpu=1, workers_per_gpu=1)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

# NOTE: This is for automatically scaling LR, USER CAN'T CHANGE THIS VALUE
mmdet_official_special_samples_per_gpu = 1
