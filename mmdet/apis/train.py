import random

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner, OptimizerHook,
                         build_optimizer, LoggerHook, load_checkpoint)

from mmdet.core import DistEvalHook, EvalHook, Fp16OptimizerHook, CompressionHook
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.utils import get_root_logger
from mmdet.parallel import MMDataCPU

from mmdet.core.nncf import wrap_nncf_model
from .fake_input import get_fake_input


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


def add_logging_on_first_and_last_iter(runner):
    def every_n_inner_iters(self, runner, n):
        if runner.inner_iter == 0 or runner.inner_iter == runner.max_iters - 1:
            return True
        return (runner.inner_iter + 1) % n == 0 if n > 0 else False

    for hook in runner.hooks:
        if isinstance(hook, LoggerHook):
            hook.every_n_inner_iters = every_n_inner_iters.__get__(hook)


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

    if cfg.load_from:
        load_checkpoint(model=model, filename=cfg.load_from)

    # put model on gpus
    if torch.cuda.is_available():
        model = model.cuda()

    # nncf model wrapper
    if cfg.nncf_enable_compression:
        compression_ctrl, model = wrap_nncf_model(model, cfg, data_loaders[0], get_fake_input)
    else:
        compression_ctrl = None

    map_location = 'default'
    if torch.cuda.is_available():
        if distributed:
            # put model on gpus
            find_unused_parameters = cfg.get('find_unused_parameters', False)
            # Sets the `find_unused_parameters` parameter in
            # torch.nn.parallel.DistributedDataParallel
            model = MMDistributedDataParallel(
                model,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = MMDataParallel(
                model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
    else:
        model = MMDataCPU(model)
        map_location = 'cpu'

    if cfg.nncf_enable_compression and distributed:
        compression_ctrl.distributed()

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

    add_logging_on_first_and_last_iter(runner)

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        if "should_evaluate_before_run" not in eval_cfg:
            eval_cfg["should_evaluate_before_run"] = cfg.nncf_enable_compression
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.nncf_enable_compression:
        runner.register_hook(CompressionHook(compression_ctrl=compression_ctrl))

    if cfg.resume_from:
        runner.resume(cfg.resume_from, map_location=map_location)

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs, compression_ctrl=compression_ctrl)
