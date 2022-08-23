_base_ = 'retinanet_pvt-t_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        num_layers=[3, 8, 27, 3],
        init_cfg=dict(checkpoint='https://github.com/whai362/PVT/'
                      'releases/download/v2/pvt_large.pth')))
# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper = dict(type='AmpOptimWrapper')
