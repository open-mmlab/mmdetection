import numpy as np
import torch

checkpoint='/home/group5/lzj/TOV_mmdetection_cache/work_dir/TinyPerson/Base/Pretrained/RESNET50_CBAM_new_name_wrap.pth'
save_checkpoint='/home/group5/lzj/TOV_mmdetection_cache/work_dir/TinyPerson/Base/Pretrained/RESNET50_CBAM_no_warp.pth'

a = torch.load(checkpoint)
b = a['state_dict']
# unexpected_key int source:
# layer0.conv1.weight, layer0.bn1.weight, layer0.bn1.bias, layer0.bn1.running_mean, layer0.bn1.running_var, last_linear.weight, last_linear.bias
# missing keys:
# conv1.weight, bn1.weight, bn1.bias, bn1.running_mean, bn1.running_var

for key in a['state_dict']:
    key_after = key[key.find('.')+1:]
    b[key_after] = a['state_dict'][key]
    del b[key]

a['state_dict'] = b
# for i in range(0,len(a['state_dict'])):
#     key = a['state_dict']


torch.save(a,save_checkpoint)
