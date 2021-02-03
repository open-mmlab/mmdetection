import argparse
import glob
import json
import os.path as osp

import mmcv


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
    parser = argparse.ArgumentParser(
        description='Gather benchmarked models metric')
    parser.add_argument(
        'root',
        type=str,
        help='root path of benchmarked models to be gathered')
    parser.add_argument(
        'benchmark_json', type=str, help='benchmark models json path')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    root_path = args.root
    benchmark_json_path = args.benchmark_json

    model_configs = mmcv.load(benchmark_json_path)['models']
    result_dict = {}
    for config in model_configs:

        config_name = osp.split(config)[-1]
        config_name = osp.splitext(config_name)[0]
        result_path = osp.join(root_path, config_name)

        if osp.exists(result_path):

            # 1 read config
            cfg = mmcv.Config.fromfile(config)
            total_epochs = cfg.total_epochs

            results_lut = cfg.evaluation.metric
            if not isinstance(results_lut, list):
                results_lut = [results_lut]

            results_lut_out = []
            for key in results_lut:
                if 'proposal_fast' in key:
                    results_lut_out.append('AR@1000')  # RPN
                elif 'mAP' not in key:
                    results_lut_out.append(key + '_mAP')

            # 2 determine whether total_epochs ckpt exists
            ckpt_path = f'epoch_{total_epochs}.pth'
            if osp.exists(osp.join(result_path, ckpt_path)):
                log_json_path = list(
                    sorted(glob.glob(osp.join(result_path, '*.log.json'))))[-1]

                # 3 read metric
                model_performance = get_final_results(log_json_path,
                                                      total_epochs,
                                                      results_lut_out)
                if model_performance is None:
                    print(f'log file error: {log_json_path}')
                    continue
                result_dict[config_name] = model_performance
            else:
                print(f'not exist: {ckpt_path}')

        else:
            print(f'not exist: {result_path}')

    # 4 print results
    print('===================================')
    for config_name, metrics in result_dict.items():
        print(config_name, metrics)
    print('===================================')
