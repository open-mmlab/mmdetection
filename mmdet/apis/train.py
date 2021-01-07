import os
import random

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer)
from mmcv.utils import build_from_cfg

from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_root_logger


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # if just swa training is performed,
    # skip building the runner for the traditional training
    if not cfg.get('only_swa_training', False):
        # build runner
        optimizer = build_optimizer(model, cfg.optimizer)
        runner = EpochBasedRunner(
            model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta)
        # an ugly workaround to make .log and .log.json filenames the same
        runner.timestamp = timestamp

        # fp16 setting
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
        elif distributed and 'type' not in cfg.optimizer_config:
            optimizer_config = OptimizerHook(**cfg.optimizer_config)
        else:
            optimizer_config = cfg.optimizer_config

        # register hooks
        runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                       cfg.checkpoint_config, cfg.log_config,
                                       cfg.get('momentum_config', None))
        if distributed:
            runner.register_hook(DistSamplerSeedHook())

        # register eval hooks
        if validate:
            # Support batch_size > 1 in validation
            val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
            if val_samples_per_gpu > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                cfg.data.val.pipeline = replace_ImageToTensor(
                    cfg.data.val.pipeline)
            val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
            val_dataloader = build_dataloader(
                val_dataset,
                samples_per_gpu=val_samples_per_gpu,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False)
            eval_cfg = cfg.get('evaluation', {})
            eval_hook = DistEvalHook if distributed else EvalHook
            runner.register_hook(
                eval_hook(val_dataloader, save_best='bbox_mAP', **eval_cfg))

        # user-defined hooks
        if cfg.get('custom_hooks', None):
            custom_hooks = cfg.custom_hooks
            assert isinstance(custom_hooks, list), \
                f'custom_hooks expect list type, but got {type(custom_hooks)}'
            for hook_cfg in cfg.custom_hooks:
                assert isinstance(hook_cfg, dict), \
                    'Each item in custom_hooks expects dict type, but got ' \
                    f'{type(hook_cfg)}'
                hook_cfg = hook_cfg.copy()
                priority = hook_cfg.pop('priority', 'NORMAL')
                hook = build_from_cfg(hook_cfg, HOOKS)
                runner.register_hook(hook, priority=priority)

        if cfg.resume_from:
            runner.resume(cfg.resume_from)
        elif cfg.load_from:
            runner.load_checkpoint(cfg.load_from)
        runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
    else:
        # if just swa training is performed, there should be a starting model
        assert cfg.swa_resume_from is not None or cfg.swa_load_from is not None

    # perform swa training
    # build swa training runner
    if not cfg.get('swa_training', False):
        return
    from mmdet.core import SWAHook
    logger.info('Start SWA training')
    swa_optimizer = build_optimizer(model, cfg.swa_optimizer)
    swa_runner = EpochBasedRunner(
        model,
        optimizer=swa_optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    swa_runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        swa_optimizer_config = Fp16OptimizerHook(
            **cfg.swa_optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.swa_optimizer_config:
        swa_optimizer_config = OptimizerHook(**cfg.swa_optimizer_config)
    else:
        swa_optimizer_config = cfg.swa_optimizer_config

    # register hooks
    swa_runner.register_training_hooks(cfg.swa_lr_config, swa_optimizer_config,
                                       cfg.swa_checkpoint_config,
                                       cfg.log_config,
                                       cfg.get('momentum_config', None))
    if distributed:
        swa_runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_hook = DistEvalHook if distributed else EvalHook
        swa_runner.register_hook(eval_hook(val_dataloader, **eval_cfg))
        swa_eval = True
        swa_eval_hook = eval_hook(
            val_dataloader, save_best='bbox_mAP', **eval_cfg)
    else:
        swa_eval = False
        swa_eval_hook = None

    # register swa hook
    swa_hook = SWAHook(swa_eval=swa_eval, eval_hook=swa_eval_hook)
    swa_runner.register_hook(swa_hook, priority='LOW')

    # register user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            swa_runner.register_hook(hook, priority=priority)

    if cfg.swa_resume_from:
        swa_runner.resume(cfg.swa_resume_from)
    elif cfg.swa_load_from:
        # use the best pretrained model as the starting model for swa training
        if cfg.swa_load_from == 'best_bbox_mAP.pth':
            best_model_path = os.path.join(cfg.work_dir, cfg.swa_load_from)
            assert os.path.exists(best_model_path)
            # avoid the best pretrained model being overwritten
            new_best_model_path = os.path.join(cfg.work_dir,
                                               'best_bbox_mAP_pretrained.pth')
            os.rename(best_model_path, new_best_model_path)
            cfg.swa_load_from = new_best_model_path
        swa_runner.load_checkpoint(cfg.swa_load_from)

    swa_runner.run(data_loaders, cfg.workflow, cfg.swa_total_epochs)
