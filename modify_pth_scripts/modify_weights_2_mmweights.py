import numpy as np
import torch

checkpoint='/home/group5/lzj/TOV_mmdetection_cache/work_dir/TinyPerson/Base/Pretrained/se_resnet50-ce0d4300.pth'
save_checkpoint='/home/group5/lzj/TOV_mmdetection_cache/work_dir/TinyPerson/Base/Pretrained/se_resnet50-ce0d4300_modified.pth'

a = torch.load(checkpoint)
# unexpected_key int source:
# layer0.conv1.weight, layer0.bn1.weight, layer0.bn1.bias, layer0.bn1.running_mean, layer0.bn1.running_var, last_linear.weight, last_linear.bias
# missing keys:
# conv1.weight, bn1.weight, bn1.bias, bn1.running_mean, bn1.running_var

a['conv1.weight'] = a['layer0.conv1.weight']
a['bn1.weight'] = a['layer0.bn1.weight']
a['bn1.bias'] = a['layer0.bn1.bias']
a['bn1.running_mean'] = a['layer0.bn1.running_mean']
a['bn1.running_var'] = a['layer0.bn1.running_var']

del a['layer0.conv1.weight']
del a['layer0.bn1.weight']
del a['layer0.bn1.bias']
del a['layer0.bn1.running_mean']
del a['layer0.bn1.running_var']


torch.save(a,save_checkpoint)
