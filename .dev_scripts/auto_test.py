import argparse
import json
import os
import traceback

import mmcv
import torch
from mmcv import ProgressBar
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


class MyProgressBar(ProgressBar):

    def update(self):
        self.completed += 1
        elapsed = self.timer.since_start()
        if elapsed > 0:
            fps = self.completed / elapsed
        else:
            fps = float('inf')
        self.fps = fps
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            msg = '\r[{{}}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s' \
                  ''.format(self.completed, self.task_num, fps,
                            int(elapsed + 0.5), eta)

            bar_width = min(self.bar_width,
                            int(self.terminal_width - len(msg)) + 2,
                            int(self.terminal_width * 0.6))
            bar_width = max(2, bar_width)
            mark_width = int(bar_width * percentage)
            bar_chars = '>' * mark_width + ' ' * (bar_width - mark_width)
            self.file.write(msg.format(bar_chars))
        else:
            self.file.write(
                'completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                    self.completed, int(elapsed + 0.5), fps))
        self.file.flush()


def single_gpu_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = MyProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results, prog_bar.fps


def parse_results(json_log):
    metrics = dict()
    json_file = open(json_log, 'rb')
    lines = json_file.readlines()
    second_last_line, last_line = lines[-2:]
    second_last_line = json.loads(second_last_line)
    metrics['Mem'] = round(second_last_line['memory'] / 1000., 1)
    last_line = json.loads(last_line)
    for k, v in last_line.items():
        if k in ('AR@100', 'bbox_mAP', 'segm_mAP'):
            metrics[k] = v
    return metrics


parser = argparse.ArgumentParser()
parser.add_argument('model_path')
args = parser.parse_args()

if os.path.exists('mmdet/v2.0_benchmark_2nd/valid_results.json'):
    print('Please remove or rename the original valid_results.json')
    exit()

#  check if all models have corresponding configs
print('parse model infos')
print('-' * 40)
model_infos = []
for model_family in os.listdir(args.model_path):
    model_family_dir = os.path.join(args.model_path, model_family)
    if not os.path.isdir(model_family_dir):
        continue
    for model in os.listdir(model_family_dir):
        model_info = dict()
        model_dir = os.path.join(model_family_dir, model)

        config = os.path.join('configs', model_family, model)

        # skip model start with '_'
        if model.startswith('_'):
            continue
        # add '.py
        if not model.endswith('.py'):
            config += '.py'
        records = os.listdir(os.path.join(model_dir))
        cpt = [r for r in records if r[-3:] == 'pth']
        assert 0 < len(cpt) <= 1, 'check {} fails'.format(model_dir)
        cpt = os.path.join(model_dir, cpt[0])

        log = [r for r in records if r[-4:] == 'json']
        assert 0 < len(log) <= 1, 'check {} fails'.format(model_dir)
        log = os.path.join(model_dir, log[0])

        eval_results = parse_results(log)
        model_info['config'] = config
        model_info['checkpoint'] = cpt
        model_info['train_results'] = eval_results
        model_infos.append(model_info)

print('collect total: {} models'.format(len(model_infos)))
print()

print('start valid models')

for i, model_info in enumerate(model_infos):
    try:
        config = model_info['config']
        cpt = model_info['checkpoint']
        print('valid [{}/{}] model'.format(i + 1, len(model_infos)))
        print('config: {}'.format(config))
        print('checkpoint: {}'.format(cpt))
        print('-' * 40)

        valid_results = dict()
        valid_results['config'] = config
        valid_results['checkpoint'] = cpt
        for k, v in model_info['train_results'].items():
            valid_results['train_{}'.format(k)] = v

        cfg = mmcv.Config.fromfile(config)
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        cfg.model.pretrained = None
        cfg.data.test.test_mode = True
        distributed = False
        cfg.data.test.ann_file = '/mnt/lustre/share/caoyuhang/val_0_10.json'
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

        model = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        model = MMDataParallel(model, device_ids=[0])
        checkpoint = load_checkpoint(model, cpt, map_location='cpu')
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES

        outputs, fps = single_gpu_test(model, data_loader)

        eval_types = []
        train_metrics = model_info['train_results']
        if 'AR@100' in train_metrics.keys():
            eval_types.append('proposal_fast')
        if 'bbox_mAP' in train_metrics.keys():
            eval_types.append('bbox')
        if 'segm_mAP' in train_metrics.keys():
            eval_types.append('segm')

        eval_results = dataset.evaluate(outputs, eval_types)
        for k, v in eval_results.items():
            if isinstance(v, float):
                v = round(v, 3)
            valid_results['valid_{}'.format(k)] = v
        valid_results['inf_speed'] = round(fps, 1)
    except Exception:
        traceback.print_exc()
        valid_results['valid'] = None
        del model
        torch.cuda.empty_cache()

    with open('valid_results.json', 'a+') as f:
        mmcv.dump(valid_results, f, file_format='json')
        f.write('\n')
    print()
