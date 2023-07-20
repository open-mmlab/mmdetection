# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
import warnings
from typing import Optional, Sequence

import torch
from mmengine.dist import (barrier, broadcast, broadcast_object_list,
                           get_dist_info, is_main_process)
from mmengine.evaluator import BaseMetric
from mmengine.utils import mkdir_or_exist


class BaseVideoMetric(BaseMetric):
    """Base class for a metric in video task.

    The metric first processes each batch of data_samples and predictions,
    and appends the processed results to the results list. Then it
    collects all results together from all ranks if distributed training
    is used. Finally, it computes the metrics of the entire dataset.

    A subclass of class:`BaseVideoMetric` should assign a meaningful value
    to the class attribute `default_prefix`. See the argument `prefix` for
    details.
    """

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for track_data_sample in data_samples:
            video_data_samples = track_data_sample['video_data_samples']
            ori_video_len = video_data_samples[0].ori_video_length
            if ori_video_len == len(video_data_samples):
                # video process
                self.process_video(video_data_samples)
            else:
                # image process
                self.process_image(video_data_samples, ori_video_len)

    def evaluate(self, size: int = 1) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """
        if len(self.results) == 0:
            warnings.warn(
                f'{self.__class__.__name__} got empty `self.results`. Please '
                'ensure that the processed results are properly added into '
                '`self.results` in `process` method.')

        results = collect_tracking_results(self.results, self.collect_device)

        if is_main_process():
            _metrics = self.compute_metrics(results)  # type: ignore
            # Add prefix to metric names
            if self.prefix:
                _metrics = {
                    '/'.join((self.prefix, k)): v
                    for k, v in _metrics.items()
                }
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)

        # reset the results list
        self.results.clear()
        return metrics[0]


def collect_tracking_results(results: list,
                             device: str = 'cpu',
                             tmpdir: Optional[str] = None) -> Optional[list]:
    """Collected results in distributed environments. different from the
    function mmengine.dist.collect_results, tracking compute metrics don't use
    paramenter size, which means length of the entire validation dataset.
    because it's equal to video num, but compute metrics need image num.

    Args:
        results (list): Result list containing result parts to be
            collected. Each item of ``result_part`` should be a picklable
            object.
        device (str): Device name. Optional values are 'cpu' and 'gpu'.
        tmpdir (str | None): Temporal directory for collected results to
            store. If set to None, it will create a temporal directory for it.
            ``tmpdir`` should be None when device is 'gpu'. Defaults to None.

    Returns:
        list or None: The collected results.
    """
    if device not in ['gpu', 'cpu']:
        raise NotImplementedError(
            f"device must be 'cpu' or 'gpu', but got {device}")

    if device == 'gpu':
        assert tmpdir is None, 'tmpdir should be None when device is "gpu"'
        raise NotImplementedError('GPU collecting has not been supported yet')
    else:
        return collect_tracking_results_cpu(results, tmpdir)


def collect_tracking_results_cpu(result_part: list,
                                 tmpdir: Optional[str] = None
                                 ) -> Optional[list]:
    """Collect results on cpu mode.

    Saves the results on different gpus to 'tmpdir' and collects them by the
    rank 0 worker.

    Args:
        result_part (list): The part of prediction results.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. If is None, use `tempfile.mkdtemp()`
            to make a temporary path. Defaults to None.

    Returns:
        list or None: The collected results.
    """
    rank, world_size = get_dist_info()
    if world_size == 1:
        return result_part

    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ), 32, dtype=torch.uint8)
        if rank == 0:
            mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8)
            dir_tensor[:len(tmpdir)] = tmpdir
        broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.numpy().tobytes().decode().rstrip()
    else:
        mkdir_or_exist(tmpdir)

    # dump the part result to the dir
    with open(osp.join(tmpdir, f'part_{rank}.pkl'), 'wb') as f:  # type: ignore
        pickle.dump(result_part, f, protocol=2)

    barrier()

    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            path = osp.join(tmpdir, f'part_{i}.pkl')  # type: ignore
            with open(path, 'rb') as f:
                part_list.extend(pickle.load(f))
        shutil.rmtree(tmpdir)
        return part_list
