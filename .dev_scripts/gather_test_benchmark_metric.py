import argparse
import glob
import os.path as osp

import mmcv
from mmcv import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Gather benchmarked models metric')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'root',
        type=str,
        help='root path of benchmarked models to be gathered')
    parser.add_argument(
        '--out', type=str, help='output path of gathered metrics to be stored')
    parser.add_argument(
        '--not-show', action='store_true', help='not show metrics')
    parser.add_argument(
        '--show-all', action='store_true', help='show all model metrics')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    root_path = args.root
    metrics_out = args.out
    result_dict = {}

    cfg = Config.fromfile(args.config)

    for model_key in cfg:
        model_infos = cfg[model_key]
        if not isinstance(model_infos, list):
            model_infos = [model_infos]
        for model_info in model_infos:
            record_metrics = model_info['metric']
            config = model_info['config'].strip()
            fname, _ = osp.splitext(osp.basename(config))
            metric_json_dir = osp.join(root_path, fname)
            if osp.exists(metric_json_dir):
                json_list = glob.glob(osp.join(metric_json_dir, '*.json'))
                if len(json_list) > 0:
                    log_json_path = list(sorted(json_list))[-1]

                    metric = mmcv.load(log_json_path)
                    if config in metric.get('config', {}):

                        new_metrics = dict()
                        for record_metric_key in record_metrics:
                            record_metric_key_bk = record_metric_key
                            old_metric = record_metrics[record_metric_key]
                            if record_metric_key == 'AR_1000':
                                record_metric_key = 'AR@1000'
                            if record_metric_key not in metric['metric']:
                                raise KeyError(
                                    'record_metric_key not exist, please '
                                    'check your config')
                            new_metric = round(
                                metric['metric'][record_metric_key] * 100, 1)
                            new_metrics[record_metric_key_bk] = new_metric

                        if args.show_all:
                            result_dict[config] = dict(
                                before=record_metrics, after=new_metrics)
                        else:
                            for record_metric_key in record_metrics:
                                old_metric = record_metrics[record_metric_key]
                                new_metric = new_metrics[record_metric_key]
                                if old_metric != new_metric:
                                    result_dict[config] = dict(
                                        before=record_metrics,
                                        after=new_metrics)
                                    break
                    else:
                        print(f'{config} not included in: {log_json_path}')
                else:
                    print(f'{config} not exist file: {metric_json_dir}')
            else:
                print(f'{config} not exist dir: {metric_json_dir}')

    if metrics_out:
        mmcv.mkdir_or_exist(metrics_out)
        mmcv.dump(result_dict,
                  osp.join(metrics_out, 'batch_test_metric_info.json'))
    if not args.not_show:
        print('===================================')
        for config_name, metrics in result_dict.items():
            print(config_name, metrics)
        print('===================================')
