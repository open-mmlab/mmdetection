import os
from argparse import ArgumentParser
from datetime import date

import mmcv
from tools.publish_model import process_checkpoint

parser = ArgumentParser('Automatically publish models in a directory')
parser.add_argument('model_root')
parser.add_argument('publish_model_root')
args = parser.parse_args()

model_root = args.model_root
publish_model_root = args.publish_model_root
for model_dir in os.listdir(args.model_root):
    # skip some system files like .DS_Store
    if model_dir.startswith('.'):
        continue
    print(f'publish {model_dir}')
    model_dir_path = os.path.join(model_root, model_dir)
    model = [f for f in os.listdir(model_dir_path) if f.endswith('.pth')]
    assert len(model) <= 2, 'Each model dir can have only one model file'
    # some models may be published, pick the model with shorter name
    if len(model) == 2:
        if len(model[0]) < len(model[1]):
            model = model[0]
        else:
            model = model[1]
    else:
        model = model[0]
    # skip published model
    # e.g. vfnet_r50_fpn_1x_coco_20201027-38db6f58.pth
    if '-' in model and len(model.split('-')[1].split('.')[0]) == 8:
        continue
    model_path = os.path.join(model_dir_path, model)
    date_str = str(date.today()).replace('-', '')
    # faster_rcnn_r50_fpn_1x_coco.pth
    model_name_split = model.split('.')
    publish_model_name = model_name_split[
        0] + f'_{date_str}' + model_name_split[1]
    publish_model_dir_path = os.path.join(publish_model_root, model_dir)
    mmcv.mkdir_or_exist(publish_model_dir_path)
    publish_model_path = os.path.join(publish_model_dir_path,
                                      publish_model_name)
    process_checkpoint(model_path, publish_model_path)
    print(f'publish {model_path} to {publish_model_path}')
