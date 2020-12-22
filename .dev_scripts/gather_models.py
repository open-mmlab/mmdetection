import argparse
import glob
import json
import os.path as osp
import shutil
import subprocess

import mmcv
import torch


def process_checkpoint(in_file, out_file):
    checkpoint = torch.load(in_file, map_location='cpu')
    # remove optimizer for smaller file size
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
    # if it is necessary to remove some sensitive data in checkpoint['meta'],
    # add the code here.
    torch.save(checkpoint, out_file)
    sha = subprocess.check_output(['sha256sum', out_file]).decode()
    final_file = out_file.rstrip('.pth') + '-{}.pth'.format(sha[:8])
    subprocess.Popen(['mv', out_file, final_file])
    return final_file


def get_final_epoch(config):
    cfg = mmcv.Config.fromfile('./configs/' + config)
    return cfg.total_epochs


def get_final_results(log_json_path, epoch, results_lut):
    result_dict = dict()
    with open(log_json_path, 'r') as f:
        for line in f.readlines():
            log_line = json.loads(line)
            if 'mode' not in log_line.keys():
                continue

            if log_line['mode'] == 'train' and log_line['epoch'] == epoch:
                result_dict['memory'] = log_line['memory']

            if log_line['mode'] == 'val' and log_line['epoch'] == epoch:
                result_dict.update({
                    key: log_line[key]
                    for key in results_lut if key in log_line
                })
                return result_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Gather benchmarked models')
    parser.add_argument(
        'root',
        type=str,
        help='root path of benchmarked models to be gathered')
    parser.add_argument(
        'out', type=str, help='output path of gathered models to be stored')

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
        # check whether the exps is finished
        final_epoch = get_final_epoch(used_config)
        final_model = 'epoch_{}.pth'.format(final_epoch)
        model_path = osp.join(exp_dir, final_model)

        # skip if the model is still training
        if not osp.exists(model_path):
            continue

        # get the latest logs
        log_json_path = list(sorted(glob.glob(osp.join(exp_dir, '*.log.json'))))[-1]
        log_txt_path = list(sorted(glob.glob(osp.join(exp_dir, '*.log'))))[-1]
        cfg = mmcv.Config.fromfile('./configs/' + used_config)
        results_lut = cfg.evaluation.metric
        if not isinstance(results_lut, list):
            results_lut = [results_lut]
        # case when using VOC, the evaluation key is only 'mAP'
        results_lut = [key+'_mAP' for key in results_lut if 'mAP' not in key]
        model_performance = get_final_results(log_json_path, final_epoch, results_lut)

        if model_performance is None:
            continue

        model_time = osp.split(log_txt_path)[-1].split('.')[0]
        model_infos.append(
            dict(
                config=used_config,
                results=model_performance,
                epochs=final_epoch,
                model_time=model_time,
                log_json_path=osp.split(log_json_path)[-1]))

    # publish model for each checkpoint
    publish_model_infos = []
    for model in model_infos:
        model_publish_dir = osp.join(models_out, model['config'].rstrip('.py'))
        mmcv.mkdir_or_exist(model_publish_dir)

        model_name = osp.split(model['config'])[-1].split('.')[0]

        model_name += '_' + model['model_time']
        publish_model_path = osp.join(model_publish_dir, model_name)
        trained_model_path = osp.join(models_root, model['config'],
                                      'epoch_{}.pth'.format(model['epochs']))

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
        target_cconfig_path = osp.split(config_path)[-1]
        shutil.copy(config_path,
                    osp.join(model_publish_dir, target_cconfig_path))

        model['model_path'] = final_model_path
        publish_model_infos.append(model)

    models = dict(models=publish_model_infos)
    print(f'Totally gathered {len(publish_model_infos)} models')
    mmcv.dump(models, osp.join(models_out, 'model_info.json'))


if __name__ == '__main__':
    main()
