import torch
import torch.nn as nn
from lightning.pytorch.utilities import grad_norm
from mmengine import OPTIM_WRAPPERS
from mmengine.optim import build_optim_wrapper, _ParamScheduler
import copy

from torchmetrics import MetricCollection

# from mmpl.registry import MODELS, METRICS
from mmdet.registry import MODELS, METRICS
import lightning.pytorch as pl
from mmengine.registry import OPTIMIZERS, PARAM_SCHEDULERS
from mmengine.model import BaseModel


@MODELS.register_module()
class BasePLer(pl.LightningModule, BaseModel):
    def __init__(
            self,
            hyperparameters,
            data_preprocessor=None,
            train_cfg=None,
            test_cfg=None,
            *args,
            **kwargs
    ):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if data_preprocessor is not None:
            if isinstance(data_preprocessor, nn.Module):
                self.data_preprocessor = data_preprocessor
            elif isinstance(data_preprocessor, dict):
                self.data_preprocessor = MODELS.build(data_preprocessor)
            else:
                raise TypeError('data_preprocessor should be a `dict` or '
                                f'`nn.Module` instance, but got '
                                f'{type(data_preprocessor)}')

        evaluator_cfg = copy.deepcopy(self.hyperparameters.get('evaluator', None))
        if evaluator_cfg is not None:
            for k, v in evaluator_cfg.items():
                metrics = []
                if isinstance(v, dict):
                    v = [v]
                if isinstance(v, list):
                    for metric_cfg in v:
                        metric = METRICS.build(metric_cfg)
                        metrics.append(metric)
                else:
                    raise TypeError('evaluator should be a `dict` or '
                                    f'`list` instance, but got '
                                    f'{type(evaluator_cfg)}')
                setattr(self, k, MetricCollection(metrics, prefix=k.split('_')[0]))

    def _set_grad(self, need_train_names: list=[], noneed_train_names: list=[]):
        for name, param in self.named_parameters():
            flag = False
            for need_train_name in need_train_names:
                if need_train_name in name:
                    flag = True
            for noneed_train_name in noneed_train_names:
                if noneed_train_name in name:
                    flag = False
            param.requires_grad_(flag)

        not_specific_names = []
        for name, param in self.named_parameters():
            flag_find = False
            for specific_name in need_train_names + noneed_train_names:
                if specific_name in name:
                    flag_find = True
            if not flag_find:
                not_specific_names.append(name)

        if self.local_rank == 0:
            not_specific_names = [x.split('.')[0] for x in not_specific_names]
            not_specific_names = set(not_specific_names)
            print(f"Turning off gradients for names: {noneed_train_names}")
            print(f"Turning on gradients for names: {need_train_names}")
            print(f"Turning off gradients for not specific names: {not_specific_names}")

    def _set_train_module(self, mode=True, need_train_names: list=[]):
        self.training = mode
        for name, module in self.named_children():
            flag = False
            for need_train_name in need_train_names:
                if need_train_name in name:
                    flag = True
            if flag:
                module.train(mode)
            else:
                module.eval()
        return self

    def configure_optimizers(self):
        optimizer_cfg = copy.deepcopy(self.hyperparameters.get('optimizer'))
        base_lr = optimizer_cfg.get('lr')
        base_wd = optimizer_cfg.get('weight_decay', None)

        sub_models = optimizer_cfg.pop('sub_model', None)
        if sub_models is None:
            optimizer_cfg['params'] = filter(lambda p: p.requires_grad, self.parameters())
            # optimizer_cfg['params'] = self.parameters()
        else:
            if isinstance(sub_models, str):
                sub_models = {sub_models: {}}
            if isinstance(sub_models, list):
                sub_models = {x: {} for x in sub_models}
            assert isinstance(sub_models, dict), f'sub_models should be a dict, but got {type(sub_models)}'
            # import ipdb; ipdb.set_trace()
            # set training parameters and lr
            for sub_model_name, value in sub_models.items():
                sub_attrs = sub_model_name.split('.')
                sub_model_ = self
                # import ipdb; ipdb.set_trace()
                for sub_attr in sub_attrs:
                    sub_model_ = getattr(sub_model_, sub_attr)
                # sub_model_ = self.trainer.strategy.model._forward_module.get_submodule(sub_model_name)
                if isinstance(sub_model_, torch.nn.Parameter):
                    # filter(lambda p: p.requires_grad, model.parameters())
                    # sub_models[sub_model_name]['params'] = filter(lambda p: p.requires_grad, [sub_model_])
                    sub_models[sub_model_name]['params'] = filter(lambda p: p.requires_grad, [sub_model_])
                else:
                    # import ipdb;ipdb.set_trace()
                    sub_models[sub_model_name]['params'] = filter(lambda p: p.requires_grad, sub_model_.parameters())
                    # sub_models[sub_model_name]['params'] = sub_model_.parameters()
                lr_mult = value.pop('lr_mult', 1.)
                sub_models[sub_model_name]['lr'] = base_lr * lr_mult
                if base_wd is not None:
                    decay_mult = value.pop('decay_mult', 1.)
                    sub_models[sub_model_name]['weight_decay'] = base_wd * decay_mult
                else:
                    raise ModuleNotFoundError(f'{sub_model_name} not in model')

            if self.local_rank == 0:
                print('All sub models:')
                for name, module in self.named_children():
                    print(name, end=', ')
                print()
                print('Needed train models:')
                for name, value in sub_models.items():
                    print(f'{name}', end=', ')
                print()

            optimizer_cfg['params'] = [value for key, value in sub_models.items()]

        optimizer = OPTIMIZERS.build(optimizer_cfg)
        if self.local_rank == 0:
            print('查看优化器参数')
            for param_group in optimizer.param_groups:
                print([value.shape for value in param_group['params']], '学习率: ', param_group['lr'])

        schedulers = copy.deepcopy(self.hyperparameters.get('param_scheduler', None))
        if schedulers is None:
            return [optimizer]
        param_schedulers = []
        total_step = self.trainer.estimated_stepping_batches
        for scheduler in schedulers:
            if isinstance(scheduler, _ParamScheduler):
                param_schedulers.append(scheduler)
            elif isinstance(scheduler, dict):
                _scheduler = copy.deepcopy(scheduler)
                param_schedulers.append(
                    PARAM_SCHEDULERS.build(
                        _scheduler,
                        default_args=dict(
                            optimizer=optimizer,
                            epoch_length=self.trainer.num_training_batches,
                        )
                    )
                )
            else:
                raise TypeError(
                    'scheduler should be a _ParamScheduler object or dict, '
                    f'but got {scheduler}')

        return [optimizer], param_schedulers

    def lr_scheduler_step(self, scheduler, metric):
        pass

    def log_grad(self, module=None) -> None:
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        if module is None:
            module = self
        norms = grad_norm(module, norm_type=2)
        max_grad = max(norms.values())
        min_gead = min(norms.values())
        self.log_dict(
            {'max_grad': max_grad, 'min_grad': min_gead},
            prog_bar=True,
            logger=True
        )

    def setup(self, stage: str) -> None:
        evaluators = ['train', 'val', 'test']
        for evaluator in evaluators:
            if hasattr(self, f'{evaluator}_evaluator'):
                if hasattr(self.trainer.datamodule, f'{evaluator}_dataset'):
                    dataset = getattr(self.trainer.datamodule, f'{evaluator}_dataset')
                    if hasattr(dataset, 'metainfo'):
                        evaluator_ = getattr(self, f'{evaluator}_evaluator')
                        for v in evaluator_.values():
                            if hasattr(v, 'dataset_meta'):
                                v.dataset_meta = dataset.metainfo

    def on_before_optimizer_step(self, optimizer) -> None:
        self.log_grad()

    def on_validation_epoch_end(self) -> None:
        self._log_eval_metrics('val')

    def on_test_epoch_end(self) -> None:
        self._log_eval_metrics('test')

    def on_train_epoch_end(self) -> None:
        self._log_eval_metrics('train')

    def _log_eval_metrics(self, stage):
        assert stage in ['train', 'val', 'test']
        if hasattr(self, f'{stage}_evaluator'):
            evaluator = getattr(self, f'{stage}_evaluator')
            metrics = evaluator.compute()
            metrics = {k.lower(): v for k, v in metrics.items()}
            keys = []
            for k, v in metrics.items():
                v = v.view(-1)
                for i, data in enumerate(v):
                    keys.append(f'{k}_{i}')
                    self.log(f'{k.lower()}_{i}', data, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            evaluator.reset()

            if hasattr(self.trainer, 'checkpoint_callback'):
                monitor = self.trainer.checkpoint_callback.monitor
                if (monitor is not None) and (monitor not in keys):
                    data = torch.tensor(0., device=self.device)
                    self.log(f'{monitor}', data, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
