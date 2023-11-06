# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
import os.path as osp
import shutil
import subprocess
import time
from collections import OrderedDict

import torch
import yaml
from mmengine.config import Config
from mmengine.fileio import dump
from mmengine.utils import digit_version, mkdir_or_exist, scandir


def ordered_yaml_dump(data, stream=None, Dumper=yaml.SafeDumper, **kwds):

    class OrderedDumper(Dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items())

    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)


def process_checkpoint(in_file, out_file):
    checkpoint = torch.load(in_file, map_location='cpu')
    # remove optimizer for smaller file size
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
    if 'ema_state_dict' in checkpoint:
        del checkpoint['ema_state_dict']

    # remove ema state_dict
    for key in list(checkpoint['state_dict']):
        if key.startswith('ema_'):
            checkpoint['state_dict'].pop(key)
        elif key.startswith('data_preprocessor'):
            checkpoint['state_dict'].pop(key)

    # if it is necessary to remove some sensitive data in checkpoint['meta'],
    # add the code here.
    if digit_version(torch.__version__) >= digit_version('1.6'):
        torch.save(checkpoint, out_file, _use_new_zipfile_serialization=False)
    else:
        torch.save(checkpoint, out_file)
    sha = subprocess.check_output(['sha256sum', out_file]).decode()
    final_file = out_file.rstrip('.pth') + '-{}.pth'.format(sha[:8])
    subprocess.Popen(['mv', out_file, final_file])
    return final_file


def is_by_epoch(config):
    cfg = Config.fromfile('./configs/' + config)
    return cfg.train_cfg.type == 'EpochBasedTrainLoop'


def get_final_epoch_or_iter(config):
    cfg = Config.fromfile('./configs/' + config)
    if cfg.train_cfg.type == 'EpochBasedTrainLoop':
        return cfg.train_cfg.max_epochs
    else:
        return cfg.train_cfg.max_iters


def get_best_epoch_or_iter(exp_dir):
    best_epoch_iter_full_path = list(
        sorted(glob.glob(osp.join(exp_dir, 'best_*.pth'))))[-1]
    best_epoch_or_iter_model_path = best_epoch_iter_full_path.split('/')[-1]
    best_epoch_or_iter = best_epoch_or_iter_model_path.\
        split('_')[-1].split('.')[0]
    return best_epoch_or_iter_model_path, int(best_epoch_or_iter)


def get_real_epoch_or_iter(config):
    cfg = Config.fromfile('./configs/' + config)
    if cfg.train_cfg.type == 'EpochBasedTrainLoop':
        epoch = cfg.train_cfg.max_epochs
        return epoch
    else:
        return cfg.train_cfg.max_iters


def get_final_results(log_json_path,
                      epoch_or_iter,
                      results_lut='coco/bbox_mAP',
                      by_epoch=True):
    result_dict = dict()
    with open(log_json_path) as f:
        r = f.readlines()[-1]
        last_metric = r.split(',')[0].split(': ')[-1].strip()
    result_dict[results_lut] = last_metric
    return result_dict


def get_dataset_name(config):
    # If there are more dataset, add here.
    name_map = dict(
        CityscapesDataset='Cityscapes',
        CocoDataset='COCO',
        CocoPanopticDataset='COCO',
        DeepFashionDataset='Deep Fashion',
        LVISV05Dataset='LVIS v0.5',
        LVISV1Dataset='LVIS v1',
        VOCDataset='Pascal VOC',
        WIDERFaceDataset='WIDER Face',
        OpenImagesDataset='OpenImagesDataset',
        OpenImagesChallengeDataset='OpenImagesChallengeDataset',
        Objects365V1Dataset='Objects365 v1',
        Objects365V2Dataset='Objects365 v2')
    cfg = Config.fromfile('./configs/' + config)
    return name_map[cfg.dataset_type]


def find_last_dir(model_dir):
    dst_times = []
    for time_stamp in os.scandir(model_dir):
        if osp.isdir(time_stamp):
            dst_time = time.mktime(
                time.strptime(time_stamp.name, '%Y%m%d_%H%M%S'))
            dst_times.append([dst_time, time_stamp.name])
    return max(dst_times, key=lambda x: x[0])[1]


def convert_model_info_to_pwc(model_infos):
    pwc_files = {}
    for model in model_infos:
        cfg_folder_name = osp.split(model['config'])[-2]
        pwc_model_info = OrderedDict()
        pwc_model_info['Name'] = osp.split(model['config'])[-1].split('.')[0]
        pwc_model_info['In Collection'] = 'Please fill in Collection name'
        pwc_model_info['Config'] = osp.join('configs', model['config'])

        # get metadata
        meta_data = OrderedDict()
        if 'epochs' in model:
            meta_data['Epochs'] = get_real_epoch_or_iter(model['config'])
        else:
            meta_data['Iterations'] = get_real_epoch_or_iter(model['config'])
        pwc_model_info['Metadata'] = meta_data

        # get dataset name
        dataset_name = get_dataset_name(model['config'])

        # get results
        results = []
        # if there are more metrics, add here.
        if 'bbox_mAP' in model['results']:
            metric = round(model['results']['bbox_mAP'] * 100, 1)
            results.append(
                OrderedDict(
                    Task='Object Detection',
                    Dataset=dataset_name,
                    Metrics={'box AP': metric}))
        if 'segm_mAP' in model['results']:
            metric = round(model['results']['segm_mAP'] * 100, 1)
            results.append(
                OrderedDict(
                    Task='Instance Segmentation',
                    Dataset=dataset_name,
                    Metrics={'mask AP': metric}))
        if 'PQ' in model['results']:
            metric = round(model['results']['PQ'], 1)
            results.append(
                OrderedDict(
                    Task='Panoptic Segmentation',
                    Dataset=dataset_name,
                    Metrics={'PQ': metric}))
        pwc_model_info['Results'] = results

        link_string = 'https://download.openmmlab.com/mmdetection/v3.0/'
        link_string += '{}/{}'.format(model['config'].rstrip('.py'),
                                      osp.split(model['model_path'])[-1])
        pwc_model_info['Weights'] = link_string
        if cfg_folder_name in pwc_files:
            pwc_files[cfg_folder_name].append(pwc_model_info)
        else:
            pwc_files[cfg_folder_name] = [pwc_model_info]
    return pwc_files


def parse_args():
    parser = argparse.ArgumentParser(description='Gather benchmarked models')
    parser.add_argument(
        'root',
        type=str,
        default='work_dirs',
        help='root path of benchmarked models to be gathered')
    parser.add_argument(
        '--out',
        type=str,
        default='gather',
        help='output path of gathered models to be stored')
    parser.add_argument(
        '--best',
        action='store_true',
        help='whether to gather the best model.')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    models_root = args.root
    models_out = args.out
    mkdir_or_exist(models_out)

    # find all models in the root directory to be gathered
    raw_configs = list(scandir('./configs', '.py', recursive=True))

    # filter configs that is not trained in the experiments dir
    used_configs = []
    for raw_config in raw_configs:
        if osp.exists(osp.join(models_root, raw_config)):
            used_configs.append(raw_config)
    print(f'Find {len(used_configs)} models to be gathered')

    # find final_ckpt and log file for trained each config
    # and parse the best performance
    model_infos = []
    for used_config in used_configs:
        exp_dir = osp.join(models_root, used_config)
        by_epoch = is_by_epoch(used_config)
        # check whether the exps is finished
        if args.best is True:
            final_model, final_epoch_or_iter = get_best_epoch_or_iter(exp_dir)
        else:
            final_epoch_or_iter = get_final_epoch_or_iter(used_config)
            final_model = '{}_{}.pth'.format('epoch' if by_epoch else 'iter',
                                             final_epoch_or_iter)

        model_path = osp.join(exp_dir, final_model)
        # skip if the model is still training
        if not osp.exists(model_path):
            continue

        # get the latest logs
        latest_exp_name = find_last_dir(exp_dir)
        latest_exp_json = osp.join(exp_dir, latest_exp_name, 'vis_data',
                                   latest_exp_name + '.json')

        model_performance = get_final_results(
            latest_exp_json, final_epoch_or_iter, by_epoch=by_epoch)

        if model_performance is None:
            continue

        model_info = dict(
            config=used_config,
            results=model_performance,
            final_model=final_model,
            latest_exp_json=latest_exp_json,
            latest_exp_name=latest_exp_name)
        model_info['epochs' if by_epoch else 'iterations'] =\
            final_epoch_or_iter
        model_infos.append(model_info)

    # publish model for each checkpoint
    publish_model_infos = []
    for model in model_infos:
        model_publish_dir = osp.join(models_out, model['config'].rstrip('.py'))
        mkdir_or_exist(model_publish_dir)

        model_name = osp.split(model['config'])[-1].split('.')[0]

        model_name += '_' + model['latest_exp_name']
        publish_model_path = osp.join(model_publish_dir, model_name)
        trained_model_path = osp.join(models_root, model['config'],
                                      model['final_model'])

        # convert model
        final_model_path = process_checkpoint(trained_model_path,
                                              publish_model_path)

        # copy log
        shutil.copy(model['latest_exp_json'],
                    osp.join(model_publish_dir, f'{model_name}.log.json'))

        # copy config to guarantee reproducibility
        config_path = model['config']
        config_path = osp.join(
            'configs',
            config_path) if 'configs' not in config_path else config_path
        target_config_path = osp.split(config_path)[-1]
        shutil.copy(config_path, osp.join(model_publish_dir,
                                          target_config_path))

        model['model_path'] = final_model_path
        publish_model_infos.append(model)

    models = dict(models=publish_model_infos)
    print(f'Totally gathered {len(publish_model_infos)} models')
    dump(models, osp.join(models_out, 'model_info.json'))

    pwc_files = convert_model_info_to_pwc(publish_model_infos)
    for name in pwc_files:
        with open(osp.join(models_out, name + '_metafile.yml'), 'w') as f:
            ordered_yaml_dump(pwc_files[name], f, encoding='utf-8')


if __name__ == '__main__':
    main()
