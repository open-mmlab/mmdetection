import copy
import os
import tempfile
import unittest

import torch
from mmengine import Config, MMLogger
from mmengine.dataset import Compose
from mmengine.model import BaseModel
from torch.utils.data import Dataset

from mmdet.registry import DATASETS, MODELS
from mmdet.utils import register_all_modules
from mmdet.utils.benchmark import (DataLoaderBenchmark, DatasetBenchmark,
                                   InferenceBenchmark)


@MODELS.register_module()
class ToyDetector(BaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        pass


@DATASETS.register_module()
class ToyDataset(Dataset):
    METAINFO = dict()  # type: ignore
    data = torch.randn(12, 2)
    label = torch.ones(12)

    def __init__(self):
        self.pipeline = Compose([lambda x: x])

    def __len__(self):
        return self.data.size(0)

    def get_data_info(self, index):
        return dict(inputs=self.data[index], data_sample=self.label[index])

    def __getitem__(self, index):
        return dict(inputs=self.data[index], data_sample=self.label[index])


@DATASETS.register_module()
class ToyFullInitDataset(Dataset):
    METAINFO = dict()  # type: ignore
    data = torch.randn(12, 2)
    label = torch.ones(12)

    def __init__(self):
        self.pipeline = Compose([lambda x: x])

    def __len__(self):
        return self.data.size(0)

    def get_data_info(self, index):
        return dict(inputs=self.data[index], data_sample=self.label[index])

    def full_init(self):
        pass

    def __getitem__(self, index):
        return dict(inputs=self.data[index], data_sample=self.label[index])


class TestInferenceBenchmark(unittest.TestCase):

    def setUp(self) -> None:
        register_all_modules()

        self.cfg = Config(
            dict(
                model=dict(type='ToyDetector'),
                test_dataloader=dict(
                    dataset=dict(type='ToyDataset'),
                    sampler=dict(type='DefaultSampler', shuffle=False),
                    batch_size=3,
                    num_workers=1),
                env_cfg=dict(dist_cfg=dict(backend='nccl'))))
        self.max_iter = 10
        self.log_interval = 5

    @unittest.skipIf(not torch.cuda.is_available(),
                     'test requires GPU and torch+cuda')
    def test_init_and_run(self):
        checkpoint_path = os.path.join(tempfile.gettempdir(), 'checkpoint.pth')
        torch.save(ToyDetector().state_dict(), checkpoint_path)

        cfg = copy.deepcopy(self.cfg)
        inference_benchmark = InferenceBenchmark(cfg, checkpoint_path, False,
                                                 False, self.max_iter,
                                                 self.log_interval)
        results = inference_benchmark.run()

        self.assertTrue(isinstance(results, dict))
        self.assertTrue('avg_fps' in results)
        self.assertTrue('fps_list' in results)
        self.assertEqual(len(results['fps_list']), 1)
        self.assertTrue(inference_benchmark.data_loader.num_workers == 0)
        self.assertTrue(inference_benchmark.data_loader.batch_size == 1)

        results = inference_benchmark.run(1)
        self.assertTrue('avg_fps' in results)
        self.assertTrue('fps_list' in results)
        self.assertEqual(len(results['fps_list']), 1)
        self.assertTrue(inference_benchmark.data_loader.num_workers == 0)
        self.assertTrue(inference_benchmark.data_loader.batch_size == 1)

        # test repeat
        results = inference_benchmark.run(3)
        self.assertTrue('avg_fps' in results)
        self.assertTrue('fps_list' in results)
        self.assertEqual(len(results['fps_list']), 3)

        # test cudnn_benchmark
        cfg = copy.deepcopy(self.cfg)
        cfg.env_cfg.cudnn_benchmark = True
        inference_benchmark = InferenceBenchmark(cfg, checkpoint_path, False,
                                                 False, self.max_iter,
                                                 self.log_interval)
        inference_benchmark.run(1)

        # test mp_cfg
        cfg = copy.deepcopy(self.cfg)
        cfg.env_cfg.cudnn_benchmark = True
        cfg.env_cfg.mp_cfg = {
            'mp_start_method': 'fork',
            'opencv_num_threads': 1
        }
        inference_benchmark = InferenceBenchmark(cfg, checkpoint_path, False,
                                                 False, self.max_iter,
                                                 self.log_interval)
        inference_benchmark.run(1)

        # test fp16
        cfg = copy.deepcopy(self.cfg)
        cfg.fp16 = True
        inference_benchmark = InferenceBenchmark(cfg, checkpoint_path, False,
                                                 False, self.max_iter,
                                                 self.log_interval)
        inference_benchmark.run(1)

        # test logger
        logger = MMLogger.get_instance(
            'mmdet', log_file='temp.log', log_level='INFO')
        inference_benchmark = InferenceBenchmark(
            cfg,
            checkpoint_path,
            False,
            False,
            self.max_iter,
            self.log_interval,
            logger=logger)
        inference_benchmark.run(1)
        self.assertTrue(os.path.exists('temp.log'))

        os.remove(checkpoint_path)
        os.remove('temp.log')


class TestDataLoaderBenchmark(unittest.TestCase):

    def setUp(self) -> None:
        register_all_modules()

        self.cfg = Config(
            dict(
                model=dict(type='ToyDetector'),
                train_dataloader=dict(
                    dataset=dict(type='ToyDataset'),
                    sampler=dict(type='DefaultSampler', shuffle=True),
                    batch_size=2,
                    num_workers=1),
                val_dataloader=dict(
                    dataset=dict(type='ToyDataset'),
                    sampler=dict(type='DefaultSampler', shuffle=False),
                    batch_size=1,
                    num_workers=2),
                test_dataloader=dict(
                    dataset=dict(type='ToyDataset'),
                    sampler=dict(type='DefaultSampler', shuffle=False),
                    batch_size=3,
                    num_workers=1),
                env_cfg=dict(dist_cfg=dict(backend='nccl'))))
        self.max_iter = 5
        self.log_interval = 1
        self.num_warmup = 1

    def test_init_and_run(self):
        cfg = copy.deepcopy(self.cfg)
        dataloader_benchmark = DataLoaderBenchmark(cfg, False, 'train',
                                                   self.max_iter,
                                                   self.log_interval,
                                                   self.num_warmup)
        results = dataloader_benchmark.run(1)
        self.assertTrue('avg_fps' in results)
        self.assertTrue('fps_list' in results)
        self.assertEqual(len(results['fps_list']), 1)
        self.assertTrue(dataloader_benchmark.data_loader.num_workers == 1)
        self.assertTrue(dataloader_benchmark.data_loader.batch_size == 2)

        # test repeat
        results = dataloader_benchmark.run(3)
        self.assertTrue('avg_fps' in results)
        self.assertTrue('fps_list' in results)
        self.assertEqual(len(results['fps_list']), 3)

        # test dataset_type input parameters error
        with self.assertRaises(AssertionError):
            DataLoaderBenchmark(cfg, False, 'training', self.max_iter,
                                self.log_interval, self.num_warmup)

        dataloader_benchmark = DataLoaderBenchmark(cfg, False, 'val',
                                                   self.max_iter,
                                                   self.log_interval,
                                                   self.num_warmup)
        self.assertTrue(dataloader_benchmark.data_loader.num_workers == 2)
        self.assertTrue(dataloader_benchmark.data_loader.batch_size == 1)

        dataloader_benchmark = DataLoaderBenchmark(cfg, False, 'test',
                                                   self.max_iter,
                                                   self.log_interval,
                                                   self.num_warmup)
        self.assertTrue(dataloader_benchmark.data_loader.num_workers == 1)
        self.assertTrue(dataloader_benchmark.data_loader.batch_size == 3)

        # test mp_cfg
        cfg = copy.deepcopy(self.cfg)
        cfg.env_cfg.mp_cfg = {
            'mp_start_method': 'fork',
            'opencv_num_threads': 1
        }
        dataloader_benchmark = DataLoaderBenchmark(cfg, False, 'train',
                                                   self.max_iter,
                                                   self.log_interval,
                                                   self.num_warmup)
        dataloader_benchmark.run(1)


class TestDatasetBenchmark(unittest.TestCase):

    def setUp(self) -> None:
        register_all_modules()

        self.cfg = Config(
            dict(
                model=dict(type='ToyDetector'),
                train_dataloader=dict(
                    dataset=dict(type='ToyDataset'),
                    sampler=dict(type='DefaultSampler', shuffle=True),
                    batch_size=2,
                    num_workers=1),
                val_dataloader=dict(
                    dataset=dict(type='ToyDataset'),
                    sampler=dict(type='DefaultSampler', shuffle=False),
                    batch_size=1,
                    num_workers=2),
                test_dataloader=dict(
                    dataset=dict(type='ToyDataset'),
                    sampler=dict(type='DefaultSampler', shuffle=False),
                    batch_size=3,
                    num_workers=1)))
        self.max_iter = 5
        self.log_interval = 1
        self.num_warmup = 1

    def test_init_and_run(self):
        cfg = copy.deepcopy(self.cfg)
        dataset_benchmark = DatasetBenchmark(cfg, 'train', self.max_iter,
                                             self.log_interval,
                                             self.num_warmup)
        results = dataset_benchmark.run(1)
        self.assertTrue('avg_fps' in results)
        self.assertTrue('fps_list' in results)
        self.assertEqual(len(results['fps_list']), 1)

        # test repeat
        results = dataset_benchmark.run(3)
        self.assertTrue('avg_fps' in results)
        self.assertTrue('fps_list' in results)
        self.assertEqual(len(results['fps_list']), 3)

        # test test dataset
        dataset_benchmark = DatasetBenchmark(cfg, 'test', self.max_iter,
                                             self.log_interval,
                                             self.num_warmup)
        dataset_benchmark.run(1)

        # test val dataset
        dataset_benchmark = DatasetBenchmark(cfg, 'val', self.max_iter,
                                             self.log_interval,
                                             self.num_warmup)
        dataset_benchmark.run(1)

        # test dataset_type input parameters error
        with self.assertRaises(AssertionError):
            DatasetBenchmark(cfg, 'training', self.max_iter, self.log_interval,
                             self.num_warmup)

        # test full_init
        cfg = copy.deepcopy(self.cfg)
        cfg.test_dataloader.dataset = dict(type='ToyFullInitDataset')
        dataset_benchmark = DatasetBenchmark(cfg, 'train', self.max_iter,
                                             self.log_interval,
                                             self.num_warmup)
        dataset_benchmark.run(1)
