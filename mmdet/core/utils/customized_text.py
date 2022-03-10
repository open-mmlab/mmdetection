# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import datetime
from collections import OrderedDict

import torch

import mmcv
from mmcv.runner import HOOKS
from mmcv.runner import TextLoggerHook


@HOOKS.register_module()
class CustomizedTextLoggerHook(TextLoggerHook):
    """Customized Text Logger hook.

    This logger prints out both lr and layer_0_lr.
        
    """
    
    def _log_info(self, log_dict, runner):
        # print exp name for users to distinguish experiments
        # at every ``interval_exp_name`` iterations and the end of each epoch
        if runner.meta is not None and 'exp_name' in runner.meta:
            if (self.every_n_iters(runner, self.interval_exp_name)) or (
                    self.by_epoch and self.end_of_epoch(runner)):
                exp_info = f'Exp name: {runner.meta["exp_name"]}'
                runner.logger.info(exp_info)

        if log_dict['mode'] == 'train':
            lr_str = {}
            for lr_type in ['lr', 'layer_0_lr']:
                if isinstance(log_dict[lr_type], dict):
                    lr_str[lr_type] = []
                    for k, val in log_dict[lr_type].items():
                        lr_str.append(f'{lr_type}_{k}: {val:.3e}')
                    lr_str[lr_type] = ' '.join(lr_str)
                else:
                    lr_str[lr_type] = f'{lr_type}: {log_dict[lr_type]:.3e}'

            # by epoch: Epoch [4][100/1000]
            # by iter:  Iter [100/100000]
            if self.by_epoch:
                log_str = f'Epoch [{log_dict["epoch"]}]' \
                          f'[{log_dict["iter"]}/{len(runner.data_loader)}]\t'
            else:
                log_str = f'Iter [{log_dict["iter"]}/{runner.max_iters}]\t'
            log_str += f'{lr_str["lr"]}, {lr_str["layer_0_lr"]}, '

            if 'time' in log_dict.keys():
                self.time_sec_tot += (log_dict['time'] * self.interval)
                time_sec_avg = self.time_sec_tot / (
                    runner.iter - self.start_iter + 1)
                eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                log_str += f'eta: {eta_str}, '
                log_str += f'time: {log_dict["time"]:.3f}, ' \
                           f'data_time: {log_dict["data_time"]:.3f}, '
                # statistic memory
                if torch.cuda.is_available():
                    log_str += f'memory: {log_dict["memory"]}, '
        else:
            # val/test time
            # here 1000 is the length of the val dataloader
            # by epoch: Epoch[val] [4][1000]
            # by iter: Iter[val] [1000]
            if self.by_epoch:
                log_str = f'Epoch({log_dict["mode"]}) ' \
                    f'[{log_dict["epoch"]}][{log_dict["iter"]}]\t'
            else:
                log_str = f'Iter({log_dict["mode"]}) [{log_dict["iter"]}]\t'

        log_items = []
        for name, val in log_dict.items():
            # TODO: resolve this hack
            # these items have been in log_str
            if name in [
                    'mode', 'Epoch', 'iter', 'lr', 'layer_0_lr', 'time', 'data_time',
                    'memory', 'epoch'
            ]:
                continue
            if isinstance(val, float):
                val = f'{val:.4f}'
            log_items.append(f'{name}: {val}')
        log_str += ', '.join(log_items)

        runner.logger.info(log_str)


    def log(self, runner):
        if 'eval_iter_num' in runner.log_buffer.output:
            # this doesn't modify runner.iter and is regardless of by_epoch
            cur_iter = runner.log_buffer.output.pop('eval_iter_num')
        else:
            cur_iter = self.get_iter(runner, inner_iter=True)

        log_dict = OrderedDict(
            mode=self.get_mode(runner),
            epoch=self.get_epoch(runner),
            iter=cur_iter)

        # record lr and layer_0_lr
        cur_lr = runner.current_lr()
        if isinstance(cur_lr, list):
            log_dict['layer_0_lr'] = min(cur_lr)
            log_dict['lr'] = max(cur_lr)
        else:
            assert isinstance(cur_lr, dict)
            log_dict['lr'], log_dict['layer_0_lr'] = {}, {}
            for k, lr_ in cur_lr.items():
                assert isinstance(lr_, list)
                log_dict['layer_0_lr'].update({k: min(lr_)})
                log_dict['lr'].update({k: max(lr_)})

        if 'time' in runner.log_buffer.output:
            # statistic memory
            if torch.cuda.is_available():
                log_dict['memory'] = self._get_max_memory(runner)

        log_dict = dict(log_dict, **runner.log_buffer.output)

        self._log_info(log_dict, runner)
        self._dump_log(log_dict, runner)
        return log_dict
