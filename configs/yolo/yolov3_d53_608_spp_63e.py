_base_ = './yolov3_d53_608_spp.py'

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,  # same as burn-in in darknet
    warmup_ratio=0.1,
    step=[50, 57])
# runtime settings
total_epochs = 63
