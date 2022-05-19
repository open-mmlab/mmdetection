import torch

from mmdet.models.necks.fpn_lkaattention_ssm import lka_FPN_ssm

if __name__ == '__main__':
    # model = torch.set_default_tensor_type('torch.cuda.FloatTensor')
    tensor = [torch.rand(size=(1, 256, 128, 160)),
              torch.rand(size=(1, 512, 64, 80)),
              torch.rand(size=(1, 1024, 32, 40)),
              torch.rand(size=(1, 2048, 16, 20))]

    model = lka_FPN_ssm(in_channels=[256, 512, 1024, 2048],
                    out_channels=256,
                    num_outs=5)

    results = model(tensor)
    for result in results:
        print(result.shape)
