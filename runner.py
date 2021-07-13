import argparse
import glob
import importlib
import os
import os.path as osp
from datetime import datetime
from subprocess import run

from mmcv import Config

from utils.export_images import export_images


def load_sc_dataset_cfg(data_cfg_path):
    if osp.isfile(data_cfg_path):
        print(f"Load SC Dataset CFG from: {data_cfg_path}")
        spec = importlib.util.spec_from_file_location('sc_datasets_cfg', data_cfg_path)
        sc_datasets_cfg = spec.loader.load_module()
        return sc_datasets_cfg.SC_DATASETS_CFG
    raise Exception(f"Could not load Dataset CFG from: {data_cfg_path}")


def update_labels(cfg, classes):
    num_classes = len(classes)
    classes = '[' + ','.join(f'"{x}"' for x in classes) + ']'
    if 'dataset' in cfg.data.train:
        update_config = f'data.train.dataset.classes={classes} '
    else:
        update_config = f'data.train.classes={classes} '

    update_config += f'data.val.classes={classes} data.test.classes={classes} '
    if hasattr(cfg.model, 'bbox_head'):
        update_config += f'model.bbox_head.num_classes={num_classes} '
    if hasattr(cfg.model, 'roi_head'):
        if 'mask_head' in cfg.model.roi_head.keys():
            update_config += f'model.roi_head.mask_head.num_classes={num_classes} '
        if 'bbox_head' in cfg.model.roi_head.keys():
            if isinstance(cfg.model.roi_head.bbox_head, list):
                for i, head in enumerate(cfg.model.roi_head.bbox_head):
                    update_config += f'model.roi_head.bbox_head.{i}.num_classes={num_classes} '
            else:
                update_config += f'model.roi_head.bbox_head.num_classes={num_classes} '
    return update_config


def collect_metric(path, metric):
    """ Collects average precision values in log file. """

    average_precisions = []
    if metric == 'bbox':
        beginning = 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = '
    else:
        beginning = 'F1 best score = '
    with open(path) as read_file:
        content = [line.strip() for line in read_file]
        for line in content:
            if line.startswith(beginning):
                average_precisions.append(float(line.replace(beginning, '')))
    return average_precisions


def collect_f1_thres(path):
    beginning = 'F1 conf thres = '
    with open(path) as read_file:
        content = [line.strip() for line in read_file]
        for line in content:
            if line.startswith(beginning):
                return float(line.replace(beginning, ''))


def calculate_train_time(work_dir):
    if not osp.exists(work_dir):
        return None

    log = [file for file in os.listdir(work_dir) if file.endswith('.log')]
    if not log:
        raise KeyError(f'{work_dir} has not log file')
    log_path = osp.join(work_dir, sorted(log)[-1])
    first_line, last_line = '', ''
    with open(log_path, 'r') as log_file:
        for line in log_file:
            if line.startswith('2021-'):
                line = line[:19]
                if first_line == '':
                    first_line = line
                else:
                    last_line = line

    FMT = '%Y-%m-%d %H:%M:%S'
    tdelta = (datetime.strptime(last_line, FMT) - datetime.strptime(first_line, FMT)).total_seconds() / 60
    return tdelta


def get_command_eval_line(subset, dataset, work_dir, data_root, metric='bbox'):
    """ Returns a command line for evaluation

    Args:
        subset: train/val/test
        dataset: dataset dictionary including annotation prefix, classes, etc
        work_dir:
        data_root:
        metric: 'bbox' or 'f1'

    Returns:

    """
    dataset_folder = osp.join(work_dir, dataset["name"])
    cfg_path = osp.join(dataset_folder, 'model.py')
    ckpt_path = osp.join(dataset_folder, 'latest.pth')

    if not osp.exists(dataset_folder):
        print(f'get_command_eval_line: {dataset_folder} does not exist')
        return ''

    if not osp.exists(ckpt_path):
        # best model wildcard search
        ckpt_path = sorted(glob.glob(osp.join(dataset_folder, '*.pth')), key=os.path.getmtime)[-1]

    if subset == 'train':
        split_update_config = f'--cfg-options ' \
                              f'data.test.ann_file={osp.join(data_root, dataset["train-ann-file"])} ' \
                              f'data.test.img_prefix={osp.join(data_root, dataset["train-data-root"])} '
    elif subset == 'val':
        split_update_config = f'--cfg-options ' \
                              f'data.test.ann_file={osp.join(data_root, dataset["val-ann-file"])} ' \
                              f'data.test.img_prefix={osp.join(data_root, dataset["val-data-root"])} '
    else:
        split_update_config = f'--cfg-options ' \
                              f'data.test.ann_file={osp.join(data_root, dataset["test-ann-file"])} ' \
                              f'data.test.img_prefix={osp.join(data_root, dataset["test-data-root"])} '
    # avoid time-concuming validation on test part which is equal to val part
    if subset == 'test' and dataset['name'] in ['kbts_fish', 'pcd', 'diopsis', 'vitens-tiled', 'wgisd1', 'wgisd5',
                                                'weed']:
        if metric != 'bbox':
            return f'cp {work_dir}/{dataset["name"]}/val_{metric} {work_dir}/{dataset["name"]}/test_{metric}'
        return f'cp {work_dir}/{dataset["name"]}/val {work_dir}/{dataset["name"]}/test'

    if metric != 'bbox':
        return f'python tools/test.py {cfg_path} {ckpt_path} ' \
               f'--eval {metric} {split_update_config} | tee {dataset_folder}/{subset}_{metric}'
    return f'python tools/test.py {cfg_path} {ckpt_path} ' \
           f'--eval bbox {split_update_config} | tee {dataset_folder}/{subset}'


def print_summarized_statistics(datasets, work_dir, metric):
    names = []
    metrics = []
    for dataset in datasets:
        names.append(dataset['name'])
        for subset in ('train', 'val', 'test'):
            try:
                if metric != 'bbox':
                    [metric_value] = collect_metric(f'{work_dir}/{dataset["name"]}/{subset}_{metric}', metric)
                elif metric == 'bbox':
                    [metric_value] = collect_metric(f'{work_dir}/{dataset["name"]}/{subset}', metric)
                metrics.append(str(metric_value))
                if metric_value is None:
                    metrics.append('')
            except Exception as e:
                print(dataset['name'], subset, str(e))
                metrics.append('')
        try:
            if metric == 'bbox':
                # append empty time, since is not currently estimated
                training_time = calculate_train_time(f'{work_dir}/{dataset["name"]}/')
                metrics.append(f'{training_time:.0f}')
            elif metric == 'f1':
                f1_thres = collect_f1_thres(f'{work_dir}/{dataset["name"]}/val_f1')
                metrics.append(f'{f1_thres:.3f}')
        except Exception as e:
            metrics.append('')

    print(work_dir)
    print(','.join(names))
    print(','.join(metrics))


def main(args, datasets, skip=None, metric='bbox'):
    config_path = osp.join(args.work_dir, 'model.py')
    cfg = Config.fromfile(config_path)
    dataset_flag = '.dataset' if 'dataset' in cfg.data.train else ''
    for dataset in datasets:
        if dataset['name'] in skip:
            continue
        label_config = update_labels(cfg, dataset['classes'])
        update_config = f'--cfg-options ' \
                        f'{label_config} ' \
                        f'data.samples_per_gpu={args.batch_size} ' \
                        f'data.train{dataset_flag}.ann_file={osp.join(args.data_root, dataset["train-ann-file"])} ' \
                        f'data.train{dataset_flag}.img_prefix={osp.join(args.data_root, dataset["train-data-root"])} ' \
                        f'data.val.ann_file={osp.join(args.data_root, dataset["val-ann-file"])} ' \
                        f'data.val.img_prefix={osp.join(args.data_root, dataset["val-data-root"])} ' \
                        f'data.test.ann_file={osp.join(args.data_root, dataset["test-ann-file"])} ' \
                        f'data.test.img_prefix={osp.join(args.data_root, dataset["test-data-root"])} '

        if cfg.load_from and osp.exists(args.pretrained_root):
            load_from = osp.join(args.pretrained_root, osp.basename(cfg.load_from))
            update_config += f'load_from={load_from} '

        log_dir = osp.join(args.work_dir, dataset["name"])
        if 'step' in dataset:
            dataset["step"] = ",".join([str(s) for s in dataset["step"]])
            update_config += f'runner.max_iters={dataset["max_iters"]} ' \
                             f'lr_config.step={dataset["step"]} '
        train_command_line = f'./tools/dist_train.sh {config_path} {args.gpus} {update_config} ' \
                             f'--work-dir {log_dir} '
        if not args.val_only:
            print(train_command_line)
            run(train_command_line, shell=True, check=True)
        for subset in ['train', 'val', 'test']:
            eval_command_line = get_command_eval_line(subset, dataset, args.work_dir, args.data_root, metric=metric)
            print(eval_command_line)
            run(eval_command_line, shell=True, check=True)
    print_summarized_statistics(datasets, args.work_dir, metric)


def parse_args():
    parser = argparse.ArgumentParser(description='Run train script multiple times to get the best model')
    parser.add_argument('--work-dir', help='work dir path')
    parser.add_argument('--data-root', type=str)
    parser.add_argument('--data-cfg', type=str, default='configs/dataset_cfg/sc_datasets_cfg.py')

    subparsers = parser.add_subparsers(dest='task', help='task parser')
    parser_plt = subparsers.add_parser('train', help='parser for training')
    parser_plt.add_argument('--val-only', action='store_true')
    parser_plt.add_argument('--gpus', type=int, default=1)
    parser_plt.add_argument('--batch-size', type=int, default=6)
    parser_plt.add_argument('--pretrained-root', type=str, default="/home/yuchunli/_MODELS/mmdet")

    parser_plt = subparsers.add_parser('export_images', help='parser for generating predicted images with ground-truth')
    parser_plt.add_argument('--font-size', type=int, default=13)
    parser_plt.add_argument('--border-width', type=int, default=2)
    parser_plt.add_argument('--score-thres', type=int, default=0.1)
    parser_plt.add_argument('--save-all', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    sc_datasets = load_sc_dataset_cfg(args.data_cfg)
    if args.task == 'train':
        main(args, sc_datasets, skip=('weed', 'diopsis'), metric='bbox')

    if args.task == 'export_images':
        export_images(sc_datasets, args.work_dir, args.data_root, export_images=False, font_size=args.font_size,
                      border_width=args.border_width, save_all=args.save_all, score_thres=args.score_thres,
                      format_only=True)

    # estimate_train_time(args, sc_datset)
