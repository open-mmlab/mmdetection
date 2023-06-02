_base_ = './faster-rcnn_r50-caffe-dc5_ms-1x_coco.py'

# MMEngine support the following two ways, users can choose
# according to convenience
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500), # noqa
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=12,
#         by_epoch=True,
#         milestones=[28, 34],
#         gamma=0.1)
# ]
_base_.param_scheduler[1].milestones = [28, 34]

train_cfg = dict(max_epochs=36)
