import argparse
import json
import os
import os.path as osp
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style('dark')

plt.switch_backend('Agg')


def cal_train_time(log_dicts, args):
    for i, log_dict in enumerate(log_dicts):
        print('{}Analyze train time of {}{}'.format('-' * 5, args.json_logs[i],
                                                    '-' * 5))
        epochs = log_dict.keys()
        all_times = []
        for epoch in epochs:
            epoch_times = log_dict[epoch]['time']
            if not args.include_outliers:
                epoch_times = epoch_times[1:]
            all_times.append(epoch_times)
        all_times = np.array(all_times)
        ave_time_over_epoch = all_times.mean(-1)
        slowest_epoch = ave_time_over_epoch.argmax()
        fastest_epoch = ave_time_over_epoch.argmin()
        std_over_epoch = ave_time_over_epoch.std()
        print('slowest epoch {:02d}, average time is {:.4f}'.format(
            slowest_epoch + 1, ave_time_over_epoch[slowest_epoch]))
        print('fastest epoch {:02d}, average time is {:.4f}'.format(
            fastest_epoch + 1, ave_time_over_epoch[fastest_epoch]))
        print('time std over epochs is {:.4f}'.format(std_over_epoch))
        print('average iter time: {:.4f} s/iter'.format(np.mean(all_times)))
        print()


def plot_curve(log_dicts, args):
    # if legend is None, use file name of json logs as legend
    legend = args.legend if args.legend is not None else args.json_logs
    assert len(legend) == len(args.json_logs)
    metric = args.key

    for i, log_dict in enumerate(log_dicts):
        print('plot curve of {}, metric is {}'.format(args.json_logs[i],
                                                      metric))
        epochs = log_dict.keys()
        assert metric in log_dict[list(epochs)
                                  [0]], '{} does not contain metric {}'.format(
                                      args.json_logs[i], metric)

        if metric in ('bbox_mAP', 'segm_mAP'):
            xs = np.arange(1, max(epochs) + 1)
            ys = []
            for epoch in epochs:
                ys += log_dict[epoch][metric]
            ax = plt.gca()
            ax.set_xticks(xs)
            plt.title('{} v.s. epoch'.format(metric))
            plt.xlabel('epoch')
            plt.ylabel(metric)
            plt.plot(xs, ys, label=legend[i], marker='o')
        elif metric in ('loss', ):
            xs = []
            ys = []
            num_iters_per_epoch = log_dict[list(epochs)[0]]['iter'][-1]
            for epoch in epochs:
                iters = log_dict[epoch]['iter']
                if log_dict[epoch]['mode'][-1] == 'val':
                    iters = iters[:-1]
                xs.append(np.array(iters) + (epoch - 1) * num_iters_per_epoch)
                ys.append(np.array(log_dict[epoch]['loss']))
            xs = np.concatenate(xs)
            ys = np.concatenate(ys)
            plt.title('{} v.s. iter'.format(metric))
            plt.xlabel('iter')
            plt.ylabel(metric)
            plt.plot(xs, ys, label=legend[i], linewidth=0.5)
        plt.legend()
    out = args.out
    if out is None:
        if not osp.exists('analysis_log_res'):
            os.mkdir('analysis_log_res')
        out = 'analysis_log_res/{}.pdf'.format(metric)
    print('save curve to: {}'.format(out))
    plt.savefig(out)
    plt.cla()


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    parser.add_argument(
        'task',
        type=str,
        choices=['plot_curve', 'cal_train_time'],
        help='currently support plot curve and calculate average train time')
    parser.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser.add_argument(
        '--key',
        type=str,
        choices=['bbox_mAP', 'segm_mAP', 'loss'],
        default='bbox_mAP',
        help='the metric that you want to plot')
    parser.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend of each plot')
    parser.add_argument(
        '--include_outliers',
        action='store_true',
        help='whether to reserve the time of first iter of every epoch')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    json_logs = args.json_logs
    for json_log in json_logs:
        assert json_log.endswith('.json')

    # convert the json log into log_dict, key is epoch, value is a sub dict
    # keys of sub dict is different metrics, e.g. memory, bbox_mAP
    # value of sub dict is a list of corresponding values of all iterations
    log_dicts = [dict() for _ in json_logs]
    for json_log, log_dict in zip(json_logs, log_dicts):
        with open(json_log, 'r') as log_file:
            for l in log_file:
                log = json.loads(l.strip())
                epoch = log.pop('epoch')
                if epoch not in log_dict:
                    log_dict[epoch] = defaultdict(list)
                for k, v in log.items():
                    log_dict[epoch][k].append(v)

    eval(args.task)(log_dicts, args)


if __name__ == '__main__':
    main()
