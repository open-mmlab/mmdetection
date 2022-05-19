from mmdet.models.necks.pafpn_unified_carafe import PAFPN_UNIFIED_CARAFE
import torch

# model = FPNRefineModule(target_level=1,
#                         input_level=4,
#                         channel=256)
# model = PAFPN_FPNRefine(
#         in_channels=[256, 512, 1024, 2048],
#         out_channels=256,
#         num_outs=5).cuda()
model = PAFPN_UNIFIED_CARAFE(in_channels=[256, 512, 1024, 2048],
                                 out_channels=256,
                                 num_outs=5)
    # .cuda()


if __name__=='__main__':
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # test_tensor = [torch.rand(size=(1, 256, 128, 160)),
    #                torch.rand(size=(1, 256, 64, 80)),
    #                torch.rand(size=(1, 256, 32, 40)),
    #                torch.rand(size=(1, 256, 16, 20))]
    test_tensor = [torch.rand(size=(1, 256, 128, 160)),
                   torch.rand(size=(1, 512, 64, 80)),
                   torch.rand(size=(1, 1024, 32, 40)),
                   torch.rand(size=(1, 2048, 16, 20))]

    result = model(test_tensor)
    for tensor in result:
        print(tensor.shape)