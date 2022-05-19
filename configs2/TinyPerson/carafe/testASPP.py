import torch

from mmdet.models.necks.ssfpn import ASPP


model = ASPP(256,16)
input = torch.randn((1,256,16,16))

output = model(input)
print(output.shape)
