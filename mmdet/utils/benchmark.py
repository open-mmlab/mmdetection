# Copyright (c) OpenMMLab. All rights reserved.
import copy
import time
from functools import partial
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import fuse_conv_bn
# TODO need update
# from mmcv.runner import wrap_fp16_model
from mmengine import MMLogger
from mmengine.config import Config
from mmengine.device import get_max_cuda_memory
from mmengine.dist import get_world_size
from mmengine.runner import Runner, load_checkpoint
from mmengine.utils.dl_utils import set_multi_processing
from torch.nn.parallel import DistributedDataParallel

from mmdet.registry import DATASETS, MODELS

try:
    import psutil
except ImportError:
    psutil = None


def custom_round(value: Union[int, float],
                 factor: Union[int, float],
                 precision: int = 2) -> float:
    """Custom round function."""
    return round(value / factor, precision)


gb_round = partial(custom_round, factor=1024**3)


def print_log(msg: str, logger: Optional[MMLogger] = None) -> None:
    """Print a log message."""
    if logger is None:
        print(msg, flush=True)
    else:
        logger.info(msg)


def print_process_memory(p: psutil.Process,
                         logger: Optional[MMLogger] = None) -> None:
    """print process memory info."""
    mem_used = gb_round(psutil.virtual_memory().used)
    memory_full_info = p.memory_full_info()
    uss_mem = gb_round(memory_full_info.uss)
    pss_mem = gb_round(memory_full_info.pss)
    for children in p.children():
        child_mem_info = children.memory_full_info()
        uss_mem += gb_round(child_mem_info.uss)
        pss_mem += gb_round(child_mem_info.pss)
    process_count = 1 + len(p.children())
    print_log(
        f'(GB) mem_used: {mem_used:.2f} | uss: {uss_mem:.2f} | '
        f'pss: {pss_mem:.2f} | total_proc: {process_count}', logger)


class BaseBenchmark:
    """The benchmark base class.

    The ``run`` method is an external calling interface, and it will
    call the ``run_once`` method ``repeat_num`` times for benchmarking.
    Finally, call the ``average_multiple_runs`` method to further process
    the results of multiple runs.

    Args:
        max_iter (int): maximum iterations of benchmark.
        log_interval (int): interval of logging.
        num_warmup (int): Number of Warmup.
        logger (MMLogger, optional): Formatted logger used to record messages.
    """

    def __init__(self,
                 max_iter: int,
                 log_interval: int,
                 num_warmup: int,
                 logger: Optional[MMLogger] = None):
        self.max_iter = max_iter
        self.log_interval = log_interval
        self.num_warmup = num_warmup
        self.logger = logger

    def run(self, repeat_num: int = 1) -> dict:
        """benchmark entry method.

        Args:
            repeat_num (int): Number of repeat benchmark.
                Defaults to 1.
        """
        assert repeat_num >= 1

        results = []
        for _ in range(repeat_num):
            results.append(self.run_once())

        results = self.average_multiple_runs(results)
        return results

    def run_once(self) -> dict:
        """Executes the benchmark once."""
        raise NotImplementedError()

    def average_multiple_runs(self, results: List[dict]) -> dict:
        """Average the results of multiple runs."""
        raise NotImplementedError()


class InferenceBenchmark(BaseBenchmark):
    """The inference benchmark class. It will be statistical inference FPS,
    CUDA memory and CPU memory information.

    Args:
        cfg (mmengine.Config): config.
        checkpoint (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``.
        distributed (bool): distributed testing flag.
        is_fuse_conv_bn (bool): Whether to fuse conv and bn, this will
            slightly increase the inference speed.
        max_iter (int): maximum iterations of benchmark. Defaults to 2000.
        log_interval (int): interval of logging. Defaults to 50.
        num_warmup (int): Number of Warmup. Defaults to 5.
        logger (MMLogger, optional): Formatted logger used to record messages.
    """

    def __init__(self,
                 cfg: Config,
                 checkpoint: str,
                 distributed: bool,
                 is_fuse_conv_bn: bool,
                 max_iter: int = 2000,
                 log_interval: int = 50,
                 num_warmup: int = 5,
                 logger: Optional[MMLogger] = None):
        super().__init__(max_iter, log_interval, num_warmup, logger)

        assert get_world_size(
        ) == 1, 'Inference benchmark does not allow distributed multi-GPU'

        self.cfg = copy.deepcopy(cfg)
        self.distributed = distributed

        if psutil is None:
            raise ImportError('psutil is not installed, please install it by: '
                              'pip install psutil')

        self._process = psutil.Process()
        env_cfg = self.cfg.get('env_cfg')
        if env_cfg.get('cudnn_benchmark'):
            torch.backends.cudnn.benchmark = True

        mp_cfg: dict = env_cfg.get('mp_cfg', {})
        set_multi_processing(**mp_cfg, distributed=self.distributed)

        print_log('before build: ', self.logger)
        print_process_memory(self._process, self.logger)

        self.model = self._init_model(checkpoint, is_fuse_conv_bn)

        # Because multiple processes will occupy additional CPU resources,
        # FPS statistics will be more unstable when num_workers is not 0.
        # It is reasonable to set num_workers to 0.
        dataloader_cfg = cfg.test_dataloader
        dataloader_cfg['num_workers'] = 0
        dataloader_cfg['batch_size'] = 1
        dataloader_cfg['persistent_workers'] = False
        self.data_loader = Runner.build_dataloader(dataloader_cfg)

        print_log('after build: ', self.logger)
        print_process_memory(self._process, self.logger)

    def _init_model(self, checkpoint: str, is_fuse_conv_bn: bool) -> nn.Module:
        """Initialize the model."""
        model = MODELS.build(self.cfg.model)
        # TODO need update
        # fp16_cfg = self.cfg.get('fp16', None)
        # if fp16_cfg is not None:
        #     wrap_fp16_model(model)

        load_checkpoint(model, checkpoint, map_location='cpu')
        if is_fuse_conv_bn:
            model = fuse_conv_bn(model)

        model = model.cuda()

        if self.distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=False)

        model.eval()
        return model

    def run_once(self) -> dict:
        """Executes the benchmark once."""
        pure_inf_time = 0
        fps = 0

        for i, data in enumerate(self.data_loader):

            if (i + 1) % self.log_interval == 0:
                print_log('==================================', self.logger)

            torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.no_grad():
                self.model.test_step(data)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            if i >= self.num_warmup:
                pure_inf_time += elapsed
                if (i + 1) % self.log_interval == 0:
                    fps = (i + 1 - self.num_warmup) / pure_inf_time
                    cuda_memory = get_max_cuda_memory()

                    print_log(
                        f'Done image [{i + 1:<3}/{self.max_iter}], '
                        f'fps: {fps:.1f} img/s, '
                        f'times per image: {1000 / fps:.1f} ms/img, '
                        f'cuda memory: {cuda_memory} MB', self.logger)
                    print_process_memory(self._process, self.logger)

            if (i + 1) == self.max_iter:
                fps = (i + 1 - self.num_warmup) / pure_inf_time
                break

        return {'fps': fps}

    def average_multiple_runs(self, results: List[dict]) -> dict:
        """Average the results of multiple runs."""
        print_log('============== Done ==================', self.logger)

        fps_list_ = [round(result['fps'], 1) for result in results]
        avg_fps_ = sum(fps_list_) / len(fps_list_)
        outputs = {'avg_fps': avg_fps_, 'fps_list': fps_list_}

        if len(fps_list_) > 1:
            times_pre_image_list_ = [
                round(1000 / result['fps'], 1) for result in results
            ]
            avg_times_pre_image_ = sum(times_pre_image_list_) / len(
                times_pre_image_list_)

            print_log(
                f'Overall fps: {fps_list_}[{avg_fps_:.1f}] img/s, '
                'times per image: '
                f'{times_pre_image_list_}[{avg_times_pre_image_:.1f}] '
                'ms/img', self.logger)
        else:
            print_log(
                f'Overall fps: {fps_list_[0]:.1f} img/s, '
                f'times per image: {1000 / fps_list_[0]:.1f} ms/img',
                self.logger)

        print_log(f'cuda memory: {get_max_cuda_memory()} MB', self.logger)
        print_process_memory(self._process, self.logger)

        return outputs


class DataLoaderBenchmark(BaseBenchmark):
    """The dataloader benchmark class. It will be statistical inference FPS and
    CPU memory information.

    Args:
        cfg (mmengine.Config): config.
        distributed (bool): distributed testing flag.
        dataset_type (str): benchmark data type, only supports ``train``,
            ``val`` and ``test``.
        max_iter (int): maximum iterations of benchmark. Defaults to 2000.
        log_interval (int): interval of logging. Defaults to 50.
        num_warmup (int): Number of Warmup. Defaults to 5.
        logger (MMLogger, optional): Formatted logger used to record messages.
    """

    def __init__(self,
                 cfg: Config,
                 distributed: bool,
                 dataset_type: str,
                 max_iter: int = 2000,
                 log_interval: int = 50,
                 num_warmup: int = 5,
                 logger: Optional[MMLogger] = None):
        super().__init__(max_iter, log_interval, num_warmup, logger)

        assert dataset_type in ['train', 'val', 'test'], \
            'dataset_type only supports train,' \
            f' val and test, but got {dataset_type}'
        assert get_world_size(
        ) == 1, 'Dataloader benchmark does not allow distributed multi-GPU'

        self.cfg = copy.deepcopy(cfg)
        self.distributed = distributed

        if psutil is None:
            raise ImportError('psutil is not installed, please install it by: '
                              'pip install psutil')
        self._process = psutil.Process()

        mp_cfg = self.cfg.get('env_cfg', {}).get('mp_cfg')
        if mp_cfg is not None:
            set_multi_processing(distributed=self.distributed, **mp_cfg)
        else:
            set_multi_processing(distributed=self.distributed)

        print_log('before build: ', self.logger)
        print_process_memory(self._process, self.logger)

        if dataset_type == 'train':
            self.data_loader = Runner.build_dataloader(cfg.train_dataloader)
        elif dataset_type == 'test':
            self.data_loader = Runner.build_dataloader(cfg.test_dataloader)
        else:
            self.data_loader = Runner.build_dataloader(cfg.val_dataloader)

        self.batch_size = self.data_loader.batch_size
        self.num_workers = self.data_loader.num_workers

        print_log('after build: ', self.logger)
        print_process_memory(self._process, self.logger)

    def run_once(self) -> dict:
        """Executes the benchmark once."""
        pure_inf_time = 0
        fps = 0

        # benchmark with 2000 image and take the average
        start_time = time.perf_counter()
        for i, data in enumerate(self.data_loader):
            elapsed = time.perf_counter() - start_time

            if (i + 1) % self.log_interval == 0:
                print_log('==================================', self.logger)

            if i >= self.num_warmup:
                pure_inf_time += elapsed
                if (i + 1) % self.log_interval == 0:
                    fps = (i + 1 - self.num_warmup) / pure_inf_time

                    print_log(
                        f'Done batch [{i + 1:<3}/{self.max_iter}], '
                        f'fps: {fps:.1f} batch/s, '
                        f'times per batch: {1000 / fps:.1f} ms/batch, '
                        f'batch size: {self.batch_size}, num_workers: '
                        f'{self.num_workers}', self.logger)
                    print_process_memory(self._process, self.logger)

            if (i + 1) == self.max_iter:
                fps = (i + 1 - self.num_warmup) / pure_inf_time
                break

            start_time = time.perf_counter()

        return {'fps': fps}

    def average_multiple_runs(self, results: List[dict]) -> dict:
        """Average the results of multiple runs."""
        print_log('============== Done ==================', self.logger)

        fps_list_ = [round(result['fps'], 1) for result in results]
        avg_fps_ = sum(fps_list_) / len(fps_list_)
        outputs = {'avg_fps': avg_fps_, 'fps_list': fps_list_}

        if len(fps_list_) > 1:
            times_pre_image_list_ = [
                round(1000 / result['fps'], 1) for result in results
            ]
            avg_times_pre_image_ = sum(times_pre_image_list_) / len(
                times_pre_image_list_)

            print_log(
                f'Overall fps: {fps_list_}[{avg_fps_:.1f}] img/s, '
                'times per batch: '
                f'{times_pre_image_list_}[{avg_times_pre_image_:.1f}] '
                f'ms/batch, batch size: {self.batch_size}, num_workers: '
                f'{self.num_workers}', self.logger)
        else:
            print_log(
                f'Overall fps: {fps_list_[0]:.1f} batch/s, '
                f'times per batch: {1000 / fps_list_[0]:.1f} ms/batch, '
                f'batch size: {self.batch_size}, num_workers: '
                f'{self.num_workers}', self.logger)

        print_process_memory(self._process, self.logger)

        return outputs


class DatasetBenchmark(BaseBenchmark):
    """The dataset benchmark class. It will be statistical inference FPS, FPS
    pre transform and CPU memory information.

    Args:
        cfg (mmengine.Config): config.
        dataset_type (str): benchmark data type, only supports ``train``,
            ``val`` and ``test``.
        max_iter (int): maximum iterations of benchmark. Defaults to 2000.
        log_interval (int): interval of logging. Defaults to 50.
        num_warmup (int): Number of Warmup. Defaults to 5.
        logger (MMLogger, optional): Formatted logger used to record messages.
    """

    def __init__(self,
                 cfg: Config,
                 dataset_type: str,
                 max_iter: int = 2000,
                 log_interval: int = 50,
                 num_warmup: int = 5,
                 logger: Optional[MMLogger] = None):
        super().__init__(max_iter, log_interval, num_warmup, logger)
        assert dataset_type in ['train', 'val', 'test'], \
            'dataset_type only supports train,' \
            f' val and test, but got {dataset_type}'
        assert get_world_size(
        ) == 1, 'Dataset benchmark does not allow distributed multi-GPU'
        self.cfg = copy.deepcopy(cfg)

        if dataset_type == 'train':
            dataloader_cfg = copy.deepcopy(cfg.train_dataloader)
        elif dataset_type == 'test':
            dataloader_cfg = copy.deepcopy(cfg.test_dataloader)
        else:
            dataloader_cfg = copy.deepcopy(cfg.val_dataloader)

        dataset_cfg = dataloader_cfg.pop('dataset')
        dataset = DATASETS.build(dataset_cfg)
        if hasattr(dataset, 'full_init'):
            dataset.full_init()
        self.dataset = dataset

    def run_once(self) -> dict:
        """Executes the benchmark once."""
        pure_inf_time = 0
        fps = 0

        total_index = list(range(len(self.dataset)))
        np.random.shuffle(total_index)

        start_time = time.perf_counter()
        for i, idx in enumerate(total_index):
            if (i + 1) % self.log_interval == 0:
                print_log('==================================', self.logger)

            get_data_info_start_time = time.perf_counter()
            data_info = self.dataset.get_data_info(idx)
            get_data_info_elapsed = time.perf_counter(
            ) - get_data_info_start_time

            if (i + 1) % self.log_interval == 0:
                print_log(f'get_data_info - {get_data_info_elapsed * 1000} ms',
                          self.logger)

            for t in self.dataset.pipeline.transforms:
                transform_start_time = time.perf_counter()
                data_info = t(data_info)
                transform_elapsed = time.perf_counter() - transform_start_time

                if (i + 1) % self.log_interval == 0:
                    print_log(
                        f'{t.__class__.__name__} - '
                        f'{transform_elapsed * 1000} ms', self.logger)

                if data_info is None:
                    break

            elapsed = time.perf_counter() - start_time

            if i >= self.num_warmup:
                pure_inf_time += elapsed
                if (i + 1) % self.log_interval == 0:
                    fps = (i + 1 - self.num_warmup) / pure_inf_time

                    print_log(
                        f'Done img [{i + 1:<3}/{self.max_iter}], '
                        f'fps: {fps:.1f} img/s, '
                        f'times per img: {1000 / fps:.1f} ms/img', self.logger)

            if (i + 1) == self.max_iter:
                fps = (i + 1 - self.num_warmup) / pure_inf_time
                break

            start_time = time.perf_counter()

        return {'fps': fps}

    def average_multiple_runs(self, results: List[dict]) -> dict:
        """Average the results of multiple runs."""
        print_log('============== Done ==================', self.logger)

        fps_list_ = [round(result['fps'], 1) for result in results]
        avg_fps_ = sum(fps_list_) / len(fps_list_)
        outputs = {'avg_fps': avg_fps_, 'fps_list': fps_list_}

        if len(fps_list_) > 1:
            times_pre_image_list_ = [
                round(1000 / result['fps'], 1) for result in results
            ]
            avg_times_pre_image_ = sum(times_pre_image_list_) / len(
                times_pre_image_list_)

            print_log(
                f'Overall fps: {fps_list_}[{avg_fps_:.1f}] img/s, '
                'times per img: '
                f'{times_pre_image_list_}[{avg_times_pre_image_:.1f}] '
                'ms/img', self.logger)
        else:
            print_log(
                f'Overall fps: {fps_list_[0]:.1f} img/s, '
                f'times per img: {1000 / fps_list_[0]:.1f} ms/img',
                self.logger)

        return outputs
