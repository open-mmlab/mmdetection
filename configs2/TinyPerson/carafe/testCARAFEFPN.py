import torch

from mmdet.models.necks.fpn_carafe import FPN_CARAFE
# from mmdet.models.utils.carafe import CARAFE
from mmdet.models.utils.carafe_plus import CARAFE
from mmcv.cnn import build_upsample_layer

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # test_tensor = [torch.rand(size=(1, 256, 128, 160)),
    #                torch.rand(size=(1, 512, 64, 80)),
    #                torch.rand(size=(1, 1024, 32, 40)),
    #                torch.rand(size=(1, 2048, 16, 20))]
    test_tensor = torch.rand((1, 16, 24, 24))
    # model = FPN_CARAFE(in_channels=[256,512,1024,2048],
    #                 out_channels=16,
    #                 num_outs=5,
    #                 start_level=0,
    #                 end_level=-1,
    #                 norm_cfg=None,
    #                 act_cfg=None,
    #                 order=('conv', 'norm', 'act'),
    #                 upsample_cfg=dict(
    #                     type='carafe_pytorch',
    #                     k_up=5,
    #                     k_enc=3,
    #                     c_mid=64),
    #                 init_cfg=None).cuda()
    # model = CARAFE(c=16,
    #                scale=0.5,
    #                k_up=5,
    #                k_enc=3,
    #                c_mid=64).cuda()
    upsample_cfg_ = dict(type='unified_carafe',
                         up_kernel=5,
                         up_group=1,
                         encoder_kernel=3,
                         encoder_dilation=1,
                         # compressed_channels=64)
                         op='downsample',
                         compressed_channels=16)
    upsample_cfg_.update(channels=16, scale_factor=2)
    model = build_upsample_layer(upsample_cfg_).cuda()

    # model = build_upsample_cfg_.update(channels=out_channels, scale_factor=2)

    result = model(test_tensor)
    for tensor in result:
        print(tensor.shape)
