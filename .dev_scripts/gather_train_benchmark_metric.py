# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os.path as osp

from gather_models import get_final_results
from mmengine.config import Config
from mmengine.fileio import dump
from mmengine.utils import mkdir_or_exist

try:
    import xlrd
except ImportError:
    xlrd = None
try:
    import xlutils
    from xlutils.copy import copy
except ImportError:
    xlutils = None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Gather benchmarked models metric')
    parser.add_argument(
        'root',
        type=str,
        help='root path of benchmarked models to be gathered')
    parser.add_argument(
        'txt_path', type=str, help='txt path output by benchmark_filter')
    parser.add_argument(
        '--out', type=str, help='output path of gathered metrics to be stored')
    parser.add_argument(
        '--not-show', action='store_true', help='not show metrics')
    parser.add_argument(
        '--excel', type=str, help='input path of excel to be recorded')
    parser.add_argument(
        '--ncol', type=int, help='Number of column to be modified or appended')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.excel:
        assert args.ncol, 'Please specify "--excel" and "--ncol" ' \
                          'at the same time'
        if xlrd is None:
            raise RuntimeError(
                'xlrd is not installed,'
                'Please use “pip install xlrd==1.2.0” to install')
        if xlutils is None:
            raise RuntimeError(
                'xlutils is not installed,'
                'Please use “pip install xlutils==2.0.0” to install')
        readbook = xlrd.open_workbook(args.excel)
        sheet = readbook.sheet_by_name('Sheet1')
        sheet_info = {}
        total_nrows = sheet.nrows
        for i in range(3, sheet.nrows):
            sheet_info[sheet.row_values(i)[0]] = i
        xlrw = copy(readbook)
        table = xlrw.get_sheet(0)

    root_path = args.root
    metrics_out = args.out

    result_dict = {}
    with open(args.txt_path, 'r') as f:
        model_cfgs = f.readlines()
        for i, config in enumerate(model_cfgs):
            config = config.strip()
            if len(config) == 0:
                continue

            config_name = osp.split(config)[-1]
            config_name = osp.splitext(config_name)[0]
            result_path = osp.join(root_path, config_name)
            if osp.exists(result_path):
                # 1 read config
                cfg = Config.fromfile(config)
                total_epochs = cfg.runner.max_epochs
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
                        sorted(glob.glob(osp.join(result_path,
                                                  '*.log.json'))))[-1]

                    # 3 read metric
                    model_performance = get_final_results(
                        log_json_path, total_epochs, final_results_out)
                    if model_performance is None:
                        print(f'log file error: {log_json_path}')
                        continue
                    for performance in model_performance:
                        if performance in ['AR@1000', 'bbox_mAP', 'segm_mAP']:
                            metric = round(
                                model_performance[performance] * 100, 1)
                            model_performance[performance] = metric
                    result_dict[config] = model_performance

                    # update and append excel content
                    if args.excel:
                        if 'AR@1000' in model_performance:
                            metrics = f'{model_performance["AR@1000"]}' \
                                      f'(AR@1000)'
                        elif 'segm_mAP' in model_performance:
                            metrics = f'{model_performance["bbox_mAP"]}/' \
                                      f'{model_performance["segm_mAP"]}'
                        else:
                            metrics = f'{model_performance["bbox_mAP"]}'

                        row_num = sheet_info.get(config, None)
                        if row_num:
                            table.write(row_num, args.ncol, metrics)
                        else:
                            table.write(total_nrows, 0, config)
                            table.write(total_nrows, args.ncol, metrics)
                            total_nrows += 1

                else:
                    print(f'{config} not exist: {ckpt_path}')
            else:
                print(f'not exist: {config}')

        # 4 save or print results
        if metrics_out:
            mkdir_or_exist(metrics_out)
            dump(result_dict, osp.join(metrics_out, 'model_metric_info.json'))
        if not args.not_show:
            print('===================================')
            for config_name, metrics in result_dict.items():
                print(config_name, metrics)
            print('===================================')
        if args.excel:
            filename, sufflx = osp.splitext(args.excel)
            xlrw.save(f'{filename}_o{sufflx}')
            print(f'>>> Output {filename}_o{sufflx}')
