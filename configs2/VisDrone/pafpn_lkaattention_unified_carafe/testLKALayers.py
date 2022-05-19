import torch

from mmdet.models.utils.lka_layer import AttentionModule


if __name__ == '__main__':
    model = AttentionModule(dim=256)
    input = torch.randn(size=(1,256,20,20))
    attn = model(input)

    print(attn.shape)


