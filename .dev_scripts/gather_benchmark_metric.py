import argparse
import glob
import os.path as osp

import mmcv
from gather_models import get_final_results


def parse_args():
    parser = argparse.ArgumentParser(
        description='Gather benchmarked models metric')
    parser.add_argument(
        'root',
        type=str,
        help='root path of benchmarked models to be gathered')
    parser.add_argument(
        'benchmark_json', type=str, help='json path of benchmark models')
    parser.add_argument(
        '--out', type=str, help='output path of gathered metrics to be stored')
    parser.add_argument(
        '--not-show', action='store_true', help='not show metrics')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    root_path = args.root
    metrics_out = args.out
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
            final_results = cfg.evaluation.metric
            if not isinstance(final_results, list):
                final_results = [final_results]
            final_results_out = []
            for key in final_results:
                if 'proposal_fast' in key:
                    final_results_out.append('AR@1000')  # RPN
                elif 'mAP' not in key:
                    final_results_out.append(key + '_mAP')

            # 2 determine whether total_epochs ckpt exists
            ckpt_path = f'epoch_{total_epochs}.pth'
            if osp.exists(osp.join(result_path, ckpt_path)):
                log_json_path = list(
                    sorted(glob.glob(osp.join(result_path, '*.log.json'))))[-1]

                # 3 read metric
                model_performance = get_final_results(log_json_path,
                                                      total_epochs,
                                                      final_results_out)
                if model_performance is None:
                    print(f'log file error: {log_json_path}')
                    continue
                result_dict[config] = model_performance
            else:
                print(f'{config} not exist: {ckpt_path}')
        else:
            print(f'not exist: {config}')

    # 4 save or print results
    if metrics_out:
        mmcv.mkdir_or_exist(metrics_out)
        mmcv.dump(result_dict, osp.join(metrics_out, 'model_metric_info.json'))
    if not args.not_show:
        print('===================================')
        for config_name, metrics in result_dict.items():
            print(config_name, metrics)
        print('===================================')
