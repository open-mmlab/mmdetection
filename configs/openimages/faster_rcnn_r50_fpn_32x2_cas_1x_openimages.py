_base_ = ['faster_rcnn_r50_fpn_32x2_1x_openimages.py']

# Use ClassAwareSampler
data = dict(
    workers_per_gpu=0,
    train_dataloader=dict(class_aware_sampler=dict(num_sample_class=1)))
