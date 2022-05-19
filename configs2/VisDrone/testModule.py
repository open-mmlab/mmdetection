# from hrfpg.get_module import get_module
from mmdet.models.necks.attention_pafpn import get_module
from mmdet.models.necks.ssfpn import ASPP
from mmdet.models.utils.global_attention_carafe import get_module
import torch



if __name__ == '__main__':
    # model = get_module()
    model = get_module(op='upsample',in_channel=16)
    # model = get_module(op='downsample',in_channel=16)
    # model = ASPP(in_channels=256,out_channels=256)

    # test_tensor = [torch.rand(size=(1, 256, 128, 160)),
    #                torch.rand(size=(1, 512, 64, 80)),
    #                torch.rand(size=(1, 1024, 32, 40)),
    #                torch.rand(size=(1, 2048, 16, 20))]
    # test_tensor = [torch.rand(size=(1, 256, 128, 160)),
    #                torch.rand(size=(1, 256, 64, 80)),
    #                torch.rand(size=(1, 256, 32, 40)),
    #                torch.rand(size=(1, 256, 16, 20))]
    lower = torch.rand(size=(1,16,10,10))
    upper = torch.rand(size=(1,16,20,20))
    # result = model(test_tensor)
    result = model(lower,upper)
    for tensor in result:
        print(tensor.shape)

