import torch
import mmcv

net = torch.load("work_dirs/knet_panoptic/knet_s3_r50_fpn_1x_coco-panoptic_20211017_151750-395fbcba.pth")
print(type(net))
state_dict = net['state_dict']
new_state = {}
for k,v in state_dict.items():
    if 'rpn_head.localization_fpn.conv_pred' in k:
        newk = k.replace('localization_fpn.',"")
    elif 'localization_fpn.aux_convs.0' in k:
        newk = k.replace('localization_fpn.aux_convs.0.','semantic_pre.')
    else:
        newk = k
    newk = newk.replace('module.','')
    for i in range(4):
        for j in range(3):
            state = 'convs_all_levels.'+str(i)+'.conv'+str(j)
            if state in k:
                newk = k.replace('convs_all_levels.'+str(i)+'.conv'+str(j),'conv_upsample_layers.'+str(i)+'.conv.'+str(j))
    new_state[newk] = v
for k,v in new_state.items():
    print(k)
net['state_dict'] = new_state
torch.save(net,"panoptic_knet.pth")
