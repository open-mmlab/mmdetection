# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import json
import os.path as osp
import shutil
import subprocess
from collections import OrderedDict

import mmcv
import torch
import yaml


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

    # remove ema state_dict
    for key in list(checkpoint['state_dict']):
        if key.startswith('ema_'):
            checkpoint['state_dict'].pop(key)

    # if it is necessary to remove some sensitive data in checkpoint['meta'],
    # add the code here.
    if torch.__version__ >= '1.6':
        torch.save(checkpoint, out_file, _use_new_zipfile_serialization=False)
    else:
        torch.save(checkpoint, out_file)
    sha = subprocess.check_output(['sha256sum', out_file]).decode()
    final_file = out_file.rstrip('.pth') + '-{}.pth'.format(sha[:8])
    subprocess.Popen(['mv', out_file, final_file])
    return final_file


def is_by_epoch(config):
    cfg = mmcv.Config.fromfile('./configs/' + config)
    return cfg.runner.type == 'EpochBasedRunner'


def get_final_epoch_or_iter(config):
    cfg = mmcv.Config.fromfile('./configs/' + config)
    if cfg.runner.type == 'EpochBasedRunner':
        return cfg.runner.max_epochs
    else:
        return cfg.runner.max_iters


def get_best_epoch_or_iter(exp_dir):
    best_epoch_iter_full_path = list(
        sorted(glob.glob(osp.join(exp_dir, 'best_*.pth'))))[-1]
    best_epoch_or_iter_model_path = best_epoch_iter_full_path.split('/')[-1]
    best_epoch_or_iter = best_epoch_or_iter_model_path.\
        split('_')[-1].split('.')[0]
    return best_epoch_or_iter_model_path, int(best_epoch_or_iter)


def get_real_epoch_or_iter(config):
    cfg = mmcv.Config.fromfile('./configs/' + config)
    if cfg.runner.type == 'EpochBasedRunner':
        epoch = cfg.runner.max_epochs
        if cfg.data.train.type == 'RepeatDataset':
            epoch *= cfg.data.train.times
        return epoch
    else:
        return cfg.runner.max_iters


def get_final_results(log_json_path,
                      epoch_or_iter,
                      results_lut,
                      by_epoch=True):
    result_dict = dict()
    last_val_line = None
    last_train_line = None
    last_val_line_idx = -1
    last_train_line_idx = -1
    with open(log_json_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            log_line = json.loads(line)
            if 'mode' not in log_line.keys():
                continue

            if by_epoch:
                if (log_line['mode'] == 'train'
                        and log_line['epoch'] == epoch_or_iter):
                    result_dict['memory'] = log_line['memory']

                if (log_line['mode'] == 'val'
                        and log_line['epoch'] == epoch_or_iter):
                    result_dict.update({
                        key: log_line[key]
                        for key in results_lut if key in log_line
                    })
                    return result_dict
            else:
                if log_line['mode'] == 'train':
                    last_train_line_idx = i
                    last_train_line = log_line

                if log_line and log_line['mode'] == 'val':
                    last_val_line_idx = i
                    last_val_line = log_line

    # bug: max_iters = 768, last_train_line['iter'] = 750
    assert last_val_line_idx == last_train_line_idx + 1, \
        'Log file is incomplete'
    result_dict['memory'] = last_train_line['memory']
    result_dict.update({
        key: last_val_line[key]
        for key in results_lut if key in last_val_line
    })

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
        OpenImagesChallengeDataset='OpenImagesChallengeDataset')
    cfg = mmcv.Config.fromfile('./configs/' + config)
    return name_map[cfg.dataset_type]


def convert_model_info_to_pwc(model_infos):
    pwc_files = {}
    for model in model_infos:
        cfg_folder_name = osp.split(model['config'])[-2]
        pwc_model_info = OrderedDict()
        pwc_model_info['Name'] = osp.split(model['config'])[-1].split('.')[0]
        pwc_model_info['In Collection'] = 'Please fill in Collection name'
        pwc_model_info['Config'] = osp.join('configs', model['config'])

        # get metadata
        memory = round(model['results']['memory'] / 1024, 1)
        meta_data = OrderedDict()
        meta_data['Training Memory (GB)'] = memory
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

        link_string = 'https://download.openmmlab.com/mmdetection/v2.0/'
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
        help='root path of benchmarked models to be gathered')
    parser.add_argument(
        'out', type=str, help='output path of gathered models to be stored')
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
    mmcv.mkdir_or_exist(models_out)

    # find all models in the root directory to be gathered
    raw_configs = list(mmcv.scandir('./configs', '.py', recursive=True))

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
        log_json_path = list(
            sorted(glob.glob(osp.join(exp_dir, '*.log.json'))))[-1]
        log_txt_path = list(sorted(glob.glob(osp.join(exp_dir, '*.log'))))[-1]
        cfg = mmcv.Config.fromfile('./configs/' + used_config)
        results_lut = cfg.evaluation.metric
        if not isinstance(results_lut, list):
            results_lut = [results_lut]
        # case when using VOC, the evaluation key is only 'mAP'
        # when using Panoptic Dataset, the evaluation key is 'PQ'.
        for i, key in enumerate(results_lut):
            if 'mAP' not in key and 'PQ' not in key:
                results_lut[i] = key + '_mAP'
        model_performance = get_final_results(log_json_path,
                                              final_epoch_or_iter, results_lut,
                                              by_epoch)

        if model_performance is None:
            continue

        model_time = osp.split(log_txt_path)[-1].split('.')[0]
        model_info = dict(
            config=used_config,
            results=model_performance,
            model_time=model_time,
            final_model=final_model,
            log_json_path=osp.split(log_json_path)[-1])
        model_info['epochs' if by_epoch else 'iterations'] =\
            final_epoch_or_iter
        model_infos.append(model_info)

    # publish model for each checkpoint
    publish_model_infos = []
    for model in model_infos:
        model_publish_dir = osp.join(models_out, model['config'].rstrip('.py'))
        mmcv.mkdir_or_exist(model_publish_dir)

        model_name = osp.split(model['config'])[-1].split('.')[0]

        model_name += '_' + model['model_time']
        publish_model_path = osp.join(model_publish_dir, model_name)
        trained_model_path = osp.join(models_root, model['config'],
                                      model['final_model'])

        # convert model
        final_model_path = process_checkpoint(trained_model_path,
                                              publish_model_path)

        # copy log
        shutil.copy(
            osp.join(models_root, model['config'], model['log_json_path']),
            osp.join(model_publish_dir, f'{model_name}.log.json'))
        shutil.copy(
            osp.join(models_root, model['config'],
                     model['log_json_path'].rstrip('.json')),
            osp.join(model_publish_dir, f'{model_name}.log'))

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
    mmcv.dump(models, osp.join(models_out, 'model_info.json'))

    pwc_files = convert_model_info_to_pwc(publish_model_infos)
    for name in pwc_files:
        with open(osp.join(models_out, name + '_metafile.yml'), 'w') as f:
            ordered_yaml_dump(pwc_files[name], f, encoding='utf-8')


if __name__ == '__main__':
    main()
