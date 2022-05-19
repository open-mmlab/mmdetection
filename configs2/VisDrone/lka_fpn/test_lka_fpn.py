import torch

from mmdet.models.necks.fpn_lkaattention import lka_FPN


if __name__ == '__main__':
    tensor = [torch.rand(size=(1, 256, 128, 160)),
              torch.rand(size=(1, 512, 64, 80)),
              torch.rand(size=(1, 1024, 32, 40)),
              torch.rand(size=(1, 2048, 16, 20))]

    model = lka_FPN(in_channels=[256, 512, 1024, 2048],
                  out_channels=256,
                  num_outs=5)

    results = model(tensor)
    for result in results:
        print(result.shape)