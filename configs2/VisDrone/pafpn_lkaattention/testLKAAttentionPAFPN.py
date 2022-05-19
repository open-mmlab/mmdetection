import torch

from mmdet.models.necks.pafpn_lkaattention import PAFPN_LKAATTENTION

if __name__ == '__main__':
    model = PAFPN_LKAATTENTION(in_channels=[256, 512, 1024, 2048],
                                              out_channels=256,
                                              num_outs=5)
    test_tensor = [torch.rand(size=(1, 256, 128, 160)),
                   torch.rand(size=(1, 512, 64, 80)),
                   torch.rand(size=(1, 1024, 32, 40)),
                   torch.rand(size=(1, 2048, 16, 20))]

    result = model(test_tensor)

    for r in result:
        print(r.shape)
