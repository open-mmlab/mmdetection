import os


import torch

# /mnt/lustre/share_data/alphatrion/model_zoo/gluon.resnet50_v1d
folder_name = 'trident'
# folder_name = 'ghm'
# folder_name = 'faster_rcnn'
# folder_name = 'mixup'
# folder_name = 'retinanet'

parti = 'mediaf'
config_path = 'configs/{}/'.format(folder_name)

for file_name in os.listdir(config_path):
    # if 'python' not in file_name:
    #     continue
    full_path = config_path + file_name
    config_prefix = file_name[:-3]
    outfile_name = config_prefix + '.txt'
    if parti == 'mediaf': parti = 'mediaf1'
    else: parti = 'mediaf'
    parti = 'mediaf'
    print(
        'nohup ./srun_scripts/%s/run.sh '
        '%s %s %s >> log/%s/%s &' %
        (folder_name, file_name, parti, file_name, folder_name, outfile_name))
