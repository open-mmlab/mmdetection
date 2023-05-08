# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import shutil
import tempfile
from collections import defaultdict
from typing import List, Optional, Union

import numpy as np
import torch

try:
    import trackeval
except ImportError:
    trackeval = None
from mmengine.dist import (all_gather_object, barrier, broadcast,
                           broadcast_object_list, get_dist_info,
                           is_main_process)
from mmengine.logging import MMLogger

from mmdet.registry import METRICS, TASK_UTILS
from .base_video_metric import BaseVideoMetric


def get_tmpdir() -> str:
    """return the same tmpdir for all processes."""
    rank, world_size = get_dist_info()
    MAX_LEN = 512
    # 32 is whitespace
    dir_tensor = torch.full((MAX_LEN, ), 32, dtype=torch.uint8)
    if rank == 0:
        tmpdir = tempfile.mkdtemp()
        tmpdir = torch.tensor(bytearray(tmpdir.encode()), dtype=torch.uint8)
        dir_tensor[:len(tmpdir)] = tmpdir
    broadcast(dir_tensor, 0)
    tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    return tmpdir


@METRICS.register_module()
class MOTChallengeMetric(BaseVideoMetric):
    """Evaluation metrics for MOT Challenge.

    Args:
        metric (str | list[str]): Metrics to be evaluated. Options are
            'HOTA', 'CLEAR', 'Identity'.
            Defaults to ['HOTA', 'CLEAR', 'Identity'].
        outfile_prefix (str, optional): Path to save the formatted results.
            Defaults to None.
        track_iou_thr (float): IoU threshold for tracking evaluation.
            Defaults to 0.5.
        benchmark (str): Benchmark to be evaluated. Defaults to 'MOT17'.
        format_only (bool): If True, only formatting the results to the
            official format and not performing evaluation. Defaults to False.
        postprocess_tracklet_cfg (List[dict], optional): configs for tracklets
            postprocessing methods. `InterpolateTracklets` is supported.
            Defaults to []
            - InterpolateTracklets:
                - min_num_frames (int, optional): The minimum length of a
                    track that will be interpolated. Defaults to 5.
                - max_num_frames (int, optional): The maximum disconnected
                    length in a track. Defaults to 20.
                - use_gsi (bool, optional): Whether to use the GSI (Gaussian-
                    smoothed interpolation) method. Defaults to False.
                - smooth_tau (int, optional): smoothing parameter in GSI.
                    Defaults to 10.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None
    Returns:
    """
    TRACKER = 'default-tracker'
    allowed_metrics = ['HOTA', 'CLEAR', 'Identity']
    allowed_benchmarks = ['MOT15', 'MOT16', 'MOT17', 'MOT20', 'DanceTrack']
    default_prefix: Optional[str] = 'motchallenge-metric'

    def __init__(self,
                 metric: Union[str, List[str]] = ['HOTA', 'CLEAR', 'Identity'],
                 outfile_prefix: Optional[str] = None,
                 track_iou_thr: float = 0.5,
                 benchmark: str = 'MOT17',
                 format_only: bool = False,
                 use_postprocess: bool = False,
                 postprocess_tracklet_cfg: Optional[List[dict]] = [],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        if trackeval is None:
            raise RuntimeError(
                'trackeval is not installed,'
                'please install it by: pip install'
                'git+https://github.com/JonathonLuiten/TrackEval.git'
                'trackeval need low version numpy, please install it'
                'by: pip install -U numpy==1.23.5')
        if isinstance(metric, list):
            metrics = metric
        elif isinstance(metric, str):
            metrics = [metric]
        else:
            raise TypeError('metric must be a list or a str.')
        for metric in metrics:
            if metric not in self.allowed_metrics:
                raise KeyError(f'metric {metric} is not supported.')
        self.metrics = metrics
        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'
        self.use_postprocess = use_postprocess
        self.postprocess_tracklet_cfg = postprocess_tracklet_cfg.copy()
        self.postprocess_tracklet_methods = [
            TASK_UTILS.build(cfg) for cfg in self.postprocess_tracklet_cfg
        ]
        assert benchmark in self.allowed_benchmarks
        self.benchmark = benchmark
        self.track_iou_thr = track_iou_thr
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir.name = get_tmpdir()
        self.seq_info = defaultdict(
            lambda: dict(seq_length=-1, gt_tracks=[], pred_tracks=[]))
        self.gt_dir = self._get_gt_dir()
        self.pred_dir = self._get_pred_dir(outfile_prefix)
        self.seqmap = osp.join(self.pred_dir, 'videoseq.txt')
        with open(self.seqmap, 'w') as f:
            f.write('name\n')

    def __del__(self):
        # To avoid tmpdir being cleaned up too early, because in multiple
        # consecutive ValLoops, the value of `self.tmp_dir.name` is unchanged,
        # and calling `tmp_dir.cleanup()` in compute_metrics will cause errors.
        self.tmp_dir.cleanup()

    def _get_pred_dir(self, outfile_prefix):
        """Get directory to save the prediction results."""
        logger: MMLogger = MMLogger.get_current_instance()

        if outfile_prefix is None:
            outfile_prefix = self.tmp_dir.name
        else:
            if osp.exists(outfile_prefix) and is_main_process():
                logger.info('remove previous results.')
                shutil.rmtree(outfile_prefix)
        pred_dir = osp.join(outfile_prefix, self.TRACKER)
        os.makedirs(pred_dir, exist_ok=True)
        return pred_dir

    def _get_gt_dir(self):
        """Get directory to save the gt files."""
        output_dir = osp.join(self.tmp_dir.name, 'gt')
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def transform_gt_and_pred(self, img_data_sample, video, frame_id):

        video = img_data_sample['img_path'].split(os.sep)[-3]
        # load gts
        if 'instances' in img_data_sample:
            gt_instances = img_data_sample['instances']
            gt_tracks = [
                np.array([
                    frame_id + 1, gt_instances[i]['instance_id'],
                    gt_instances[i]['bbox'][0], gt_instances[i]['bbox'][1],
                    gt_instances[i]['bbox'][2] - gt_instances[i]['bbox'][0],
                    gt_instances[i]['bbox'][3] - gt_instances[i]['bbox'][1],
                    gt_instances[i]['mot_conf'],
                    gt_instances[i]['category_id'],
                    gt_instances[i]['visibility']
                ]) for i in range(len(gt_instances))
            ]
            self.seq_info[video]['gt_tracks'].extend(gt_tracks)

        # load predictions
        assert 'pred_track_instances' in img_data_sample
        if self.use_postprocess:
            pred_instances = img_data_sample['pred_track_instances']
            pred_tracks = [
                pred_instances['bboxes'][i]
                for i in range(len(pred_instances['bboxes']))
            ]
        else:
            pred_instances = img_data_sample['pred_track_instances']
            pred_tracks = [
                np.array([
                    frame_id + 1, pred_instances['instances_id'][i].cpu(),
                    pred_instances['bboxes'][i][0].cpu(),
                    pred_instances['bboxes'][i][1].cpu(),
                    (pred_instances['bboxes'][i][2] -
                     pred_instances['bboxes'][i][0]).cpu(),
                    (pred_instances['bboxes'][i][3] -
                     pred_instances['bboxes'][i][1]).cpu(),
                    pred_instances['scores'][i].cpu()
                ]) for i in range(len(pred_instances['instances_id']))
            ]
        self.seq_info[video]['pred_tracks'].extend(pred_tracks)

    def process_image(self, data_samples, video_len):

        img_data_sample = data_samples[0].to_dict()
        video = img_data_sample['img_path'].split(os.sep)[-3]
        frame_id = img_data_sample['frame_id']
        if self.seq_info[video]['seq_length'] == -1:
            self.seq_info[video]['seq_length'] = video_len
        self.transform_gt_and_pred(img_data_sample, video, frame_id)

        if frame_id == video_len - 1:
            # postprocessing
            if self.postprocess_tracklet_cfg:
                info = self.seq_info[video]
                pred_tracks = np.array(info['pred_tracks'])
                for postprocess_tracklet_methods in \
                        self.postprocess_tracklet_methods:
                    pred_tracks = postprocess_tracklet_methods\
                        .forward(pred_tracks)
                info['pred_tracks'] = pred_tracks
            self._save_one_video_gts_preds(video)

    def process_video(self, data_samples):

        video_len = len(data_samples)
        for frame_id in range(video_len):
            img_data_sample = data_samples[frame_id].to_dict()
            # load basic info
            video = img_data_sample['img_path'].split(os.sep)[-3]
            if self.seq_info[video]['seq_length'] == -1:
                self.seq_info[video]['seq_length'] = video_len
            self.transform_gt_and_pred(img_data_sample, video, frame_id)

        if self.postprocess_tracklet_cfg:
            info = self.seq_info[video]
            pred_tracks = np.array(info['pred_tracks'])
            for postprocess_tracklet_methods in \
                    self.postprocess_tracklet_methods:
                pred_tracks = postprocess_tracklet_methods \
                    .forward(pred_tracks)
            info['pred_tracks'] = pred_tracks
        self._save_one_video_gts_preds(video)

    def _save_one_video_gts_preds(self, seq: str) -> None:
        """Save the gt and prediction results."""
        info = self.seq_info[seq]
        # save predictions
        pred_file = osp.join(self.pred_dir, seq + '.txt')

        pred_tracks = np.array(info['pred_tracks'])

        with open(pred_file, 'wt') as f:
            for tracks in pred_tracks:
                line = '%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,-1,-1,-1\n' % (
                    tracks[0], tracks[1], tracks[2], tracks[3], tracks[4],
                    tracks[5], tracks[6])
                f.writelines(line)

        info['pred_tracks'] = []
        # save gts
        if info['gt_tracks']:
            gt_file = osp.join(self.gt_dir, seq + '.txt')
            with open(gt_file, 'wt') as f:
                for tracks in info['gt_tracks']:
                    line = '%d,%d,%d,%d,%d,%d,%d,%d,%.5f\n' % (
                        tracks[0], tracks[1], tracks[2], tracks[3], tracks[4],
                        tracks[5], tracks[6], tracks[7], tracks[8])
                    f.writelines(line)
            info['gt_tracks'].clear()
        # save seq info
        with open(self.seqmap, 'a') as f:
            f.write(seq + '\n')
            f.close()

    def compute_metrics(self, results: list = None) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.
                Defaults to None.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # NOTICE: don't access `self.results` from the method.
        eval_results = dict()

        if self.format_only:
            return eval_results

        eval_config = trackeval.Evaluator.get_default_eval_config()

        # need to split out the tracker name
        # caused by the implementation of TrackEval
        pred_dir_tmp = self.pred_dir.rsplit(osp.sep, 1)[0]
        dataset_config = self.get_dataset_cfg(self.gt_dir, pred_dir_tmp)

        evaluator = trackeval.Evaluator(eval_config)
        dataset = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
        metrics = [
            getattr(trackeval.metrics,
                    metric)(dict(METRICS=[metric], THRESHOLD=0.5))
            for metric in self.metrics
        ]
        output_res, _ = evaluator.evaluate(dataset, metrics)
        output_res = output_res['MotChallenge2DBox'][
            self.TRACKER]['COMBINED_SEQ']['pedestrian']

        if 'HOTA' in self.metrics:
            logger.info('Evaluating HOTA Metrics...')
            eval_results['HOTA'] = np.average(output_res['HOTA']['HOTA'])
            eval_results['AssA'] = np.average(output_res['HOTA']['AssA'])
            eval_results['DetA'] = np.average(output_res['HOTA']['DetA'])

        if 'CLEAR' in self.metrics:
            logger.info('Evaluating CLEAR Metrics...')
            eval_results['MOTA'] = np.average(output_res['CLEAR']['MOTA'])
            eval_results['MOTP'] = np.average(output_res['CLEAR']['MOTP'])
            eval_results['IDSW'] = np.average(output_res['CLEAR']['IDSW'])
            eval_results['TP'] = np.average(output_res['CLEAR']['CLR_TP'])
            eval_results['FP'] = np.average(output_res['CLEAR']['CLR_FP'])
            eval_results['FN'] = np.average(output_res['CLEAR']['CLR_FN'])
            eval_results['Frag'] = np.average(output_res['CLEAR']['Frag'])
            eval_results['MT'] = np.average(output_res['CLEAR']['MT'])
            eval_results['ML'] = np.average(output_res['CLEAR']['ML'])

        if 'Identity' in self.metrics:
            logger.info('Evaluating Identity Metrics...')
            eval_results['IDF1'] = np.average(output_res['Identity']['IDF1'])
            eval_results['IDTP'] = np.average(output_res['Identity']['IDTP'])
            eval_results['IDFN'] = np.average(output_res['Identity']['IDFN'])
            eval_results['IDFP'] = np.average(output_res['Identity']['IDFP'])
            eval_results['IDP'] = np.average(output_res['Identity']['IDP'])
            eval_results['IDR'] = np.average(output_res['Identity']['IDR'])

        return eval_results

    def evaluate(self, size: int = 1) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset.
                Defaults to None.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """
        # wait for all processes to complete prediction.
        barrier()

        # gather seq_info and convert the list of dict to a dict.
        # convert self.seq_info to dict first to make it picklable.
        gathered_seq_info = all_gather_object(dict(self.seq_info))
        all_seq_info = dict()
        for _seq_info in gathered_seq_info:
            all_seq_info.update(_seq_info)
        self.seq_info = all_seq_info

        if is_main_process():
            _metrics = self.compute_metrics()  # type: ignore
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

    def get_dataset_cfg(self, gt_folder: str, tracker_folder: str):
        """Get default configs for trackeval.datasets.MotChallenge2DBox.

        Args:
            gt_folder (str): the name of the GT folder
            tracker_folder (str): the name of the tracker folder

        Returns:
            Dataset Configs for MotChallenge2DBox.
        """
        dataset_config = dict(
            # Location of GT data
            GT_FOLDER=gt_folder,
            # Trackers location
            TRACKERS_FOLDER=tracker_folder,
            # Where to save eval results
            # (if None, same as TRACKERS_FOLDER)
            OUTPUT_FOLDER=None,
            # Use self.TRACKER as the default tracker
            TRACKERS_TO_EVAL=[self.TRACKER],
            # Option values: ['pedestrian']
            CLASSES_TO_EVAL=['pedestrian'],
            # Option Values: 'MOT15', 'MOT16', 'MOT17', 'MOT20', 'DanceTrack'
            BENCHMARK=self.benchmark,
            # Option Values: 'train', 'test'
            SPLIT_TO_EVAL='val' if self.benchmark == 'DanceTrack' else 'train',
            # Whether tracker input files are zipped
            INPUT_AS_ZIP=False,
            # Whether to print current config
            PRINT_CONFIG=True,
            # Whether to perform preprocessing
            # (never done for MOT15)
            DO_PREPROC=False if self.benchmark == 'MOT15' else True,
            # Tracker files are in
            # TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            TRACKER_SUB_FOLDER='',
            # Output files are saved in
            # OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            OUTPUT_SUB_FOLDER='',
            # Names of trackers to display
            # (if None: TRACKERS_TO_EVAL)
            TRACKER_DISPLAY_NAMES=None,
            # Where seqmaps are found
            # (if None: GT_FOLDER/seqmaps)
            SEQMAP_FOLDER=None,
            # Directly specify seqmap file
            # (if none use seqmap_folder/benchmark-split_to_eval)
            SEQMAP_FILE=self.seqmap,
            # If not None, specify sequences to eval
            # and their number of timesteps
            SEQ_INFO={
                seq: info['seq_length']
                for seq, info in self.seq_info.items()
            },
            # '{gt_folder}/{seq}.txt'
            GT_LOC_FORMAT='{gt_folder}/{seq}.txt',
            # If False, data is in GT_FOLDER/BENCHMARK-SPLIT_TO_EVAL/ and in
            # TRACKERS_FOLDER/BENCHMARK-SPLIT_TO_EVAL/tracker/
            # If True, the middle 'benchmark-split' folder is skipped for both.
            SKIP_SPLIT_FOL=True,
        )

        return dataset_config
