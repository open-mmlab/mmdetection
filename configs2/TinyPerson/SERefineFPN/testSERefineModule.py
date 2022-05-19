from mmdet.models.necks.SERefineFPN import SERefineFPN
import torch


test_tensor = [torch.rand(size=(1,256,128,160)),
               torch.rand(size=(1,512,64,80)),
               torch.rand(size=(1,1024,32,40)),
               torch.rand(size=(1,2048,16,20)),]
# print(test_tensor)
module = SERefineFPN(in_channels=[256, 512, 1024, 2048],
                     out_channels=256,
                     num_outs=5,
                     target_stage=1)

results = module(test_tensor)

module.print_list_tensor_shape(results)
