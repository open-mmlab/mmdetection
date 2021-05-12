import argparse
import glob
import os.path as osp

import mmcv


def parse_args():
    parser = argparse.ArgumentParser(
        description='Gather benchmarked models metric')
    parser.add_argument(
        'root',
        type=str,
        help='root path of benchmarked models to be gathered')
    parser.add_argument(
        'text', type=str, help='Model list files that need to be batch tested')
    parser.add_argument(
        '--out', type=str, help='output path of gathered metrics to be stored')
    parser.add_argument(
        '--show-all', action='store_true', help='show all model metrics')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    root_path = args.root
    metrics_out = args.out
    result_dict = {}

    with open(args.text, 'r') as f:
        config_info = f.readlines()
        for i, config_str in enumerate(config_info):
            if len(config_str.strip()) == 0:
                continue
            config, ckpt, old_metric = config_str.split(' ')
            old_metric = float(old_metric.strip())
            fname, _ = osp.splitext(osp.basename(config))
            metric_json_dir = osp.join(root_path, fname)
            if osp.exists(metric_json_dir):
                json_list = glob.glob(osp.join(metric_json_dir, '*.json'))
                if len(json_list) > 0:
                    log_json_path = list(sorted(json_list))[-1]

                    metric = mmcv.load(log_json_path)
                    if config in metric:
                        try:
                            map = metric[config]['bbox_mAP']
                        except Exception:
                            map = metric[config]['AR@1000']
                        if args.show_all:
                            result_dict[config] = [old_metric, round(map * 100, 1)]
                        else:
                            if round(map * 100, 1) != old_metric:
                                result_dict[config] = [
                                    old_metric, round(map * 100, 1)
                                ]
                    else:
                        print(f'{config} not included in: {log_json_path}')
                else:
                    print(f'{config} not exist file: {metric_json_dir}')
            else:
                print(f'{config} not exist dir: {metric_json_dir}')

    print('===================================')
    for config_name, metrics in result_dict.items():
        print(config_name, metrics)
    print('===================================')
