# Copyright (c) OpenMMLab. All rights reserved.
"""Optimize anchor settings on a specific dataset.

This script provides two method to optimize YOLO anchors including k-means
anchor cluster and differential evolution. You can use ``--algorithm k-means``
and ``--algorithm differential_evolution`` to switch two method.

Example:
    Use k-means anchor cluster::

        python tools/analysis_tools/optimize_anchors.py ${CONFIG} \
        --algorithm k-means --input-shape ${INPUT_SHAPE [WIDTH HEIGHT]} \
        --output-dir ${OUTPUT_DIR}
    Use differential evolution to optimize anchors::

        python tools/analysis_tools/optimize_anchors.py ${CONFIG} \
        --algorithm differential_evolution \
        --input-shape ${INPUT_SHAPE [WIDTH HEIGHT]} \
        --output-dir ${OUTPUT_DIR}
"""
import argparse
import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv import Config
from scipy.optimize import differential_evolution

from mmdet.core import bbox_cxcywh_to_xyxy, bbox_overlaps, bbox_xyxy_to_cxcywh
from mmdet.datasets import build_dataset
from mmdet.utils import get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Optimize anchor parameters.')
    parser.add_argument('config', help='Train config file path.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for calculating.')
    parser.add_argument(
        '--input-shape',
        type=int,
        nargs='+',
        default=[608, 608],
        help='input image size')
    parser.add_argument(
        '--algorithm',
        default='differential_evolution',
        help='Algorithm used for anchor optimizing.'
        'Support k-means and differential_evolution for YOLO.')
    parser.add_argument(
        '--iters',
        default=1000,
        type=int,
        help='Maximum iterations for optimizer.')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='Path to save anchor optimize result.')

    args = parser.parse_args()
    return args


class BaseAnchorOptimizer:
    """Base class for anchor optimizer.

    Args:
        dataset (obj:`Dataset`): Dataset object.
        input_shape (list[int]): Input image shape of the model.
            Format in [width, height].
        logger (obj:`logging.Logger`): The logger for logging.
        device (str, optional): Device used for calculating.
            Default: 'cuda:0'
        out_dir (str, optional): Path to save anchor optimize result.
            Default: None
    """

    def __init__(self,
                 dataset,
                 input_shape,
                 logger,
                 device='cuda:0',
                 out_dir=None):
        self.dataset = dataset
        self.input_shape = input_shape
        self.logger = logger
        self.device = device
        self.out_dir = out_dir
        bbox_whs, img_shapes = self.get_whs_and_shapes()
        ratios = img_shapes.max(1, keepdims=True) / np.array([input_shape])

        # resize to input shape
        self.bbox_whs = bbox_whs / ratios

    def get_whs_and_shapes(self):
        """Get widths and heights of bboxes and shapes of images.

        Returns:
            tuple[np.ndarray]: Array of bbox shapes and array of image
            shapes with shape (num_bboxes, 2) in [width, height] format.
        """
        self.logger.info('Collecting bboxes from annotation...')
        bbox_whs = []
        img_shapes = []
        prog_bar = mmcv.ProgressBar(len(self.dataset))
        for idx in range(len(self.dataset)):
            ann = self.dataset.get_ann_info(idx)
            data_info = self.dataset.data_infos[idx]
            img_shape = np.array([data_info['width'], data_info['height']])
            gt_bboxes = ann['bboxes']
            for bbox in gt_bboxes:
                wh = bbox[2:4] - bbox[0:2]
                img_shapes.append(img_shape)
                bbox_whs.append(wh)
            prog_bar.update()
        print('\n')
        bbox_whs = np.array(bbox_whs)
        img_shapes = np.array(img_shapes)
        self.logger.info(f'Collected {bbox_whs.shape[0]} bboxes.')
        return bbox_whs, img_shapes

    def get_zero_center_bbox_tensor(self):
        """Get a tensor of bboxes centered at (0, 0).

        Returns:
            Tensor: Tensor of bboxes with shape (num_bboxes, 4)
            in [xmin, ymin, xmax, ymax] format.
        """
        whs = torch.from_numpy(self.bbox_whs).to(
            self.device, dtype=torch.float32)
        bboxes = bbox_cxcywh_to_xyxy(
            torch.cat([torch.zeros_like(whs), whs], dim=1))
        return bboxes

    def optimize(self):
        raise NotImplementedError

    def save_result(self, anchors, path=None):
        anchor_results = []
        for w, h in anchors:
            anchor_results.append([round(w), round(h)])
        self.logger.info(f'Anchor optimize result:{anchor_results}')
        if path:
            json_path = osp.join(path, 'anchor_optimize_result.json')
            mmcv.dump(anchor_results, json_path)
            self.logger.info(f'Result saved in {json_path}')


class YOLOKMeansAnchorOptimizer(BaseAnchorOptimizer):
    r"""YOLO anchor optimizer using k-means. Code refer to `AlexeyAB/darknet.
    <https://github.com/AlexeyAB/darknet/blob/master/src/detector.c>`_.

    Args:
        num_anchors (int) : Number of anchors.
        iters (int): Maximum iterations for k-means.
    """

    def __init__(self, num_anchors, iters, **kwargs):

        super(YOLOKMeansAnchorOptimizer, self).__init__(**kwargs)
        self.num_anchors = num_anchors
        self.iters = iters

    def optimize(self):
        anchors = self.kmeans_anchors()
        self.save_result(anchors, self.out_dir)

    def kmeans_anchors(self):
        self.logger.info(
            f'Start cluster {self.num_anchors} YOLO anchors with K-means...')
        bboxes = self.get_zero_center_bbox_tensor()
        cluster_center_idx = torch.randint(
            0, bboxes.shape[0], (self.num_anchors, )).to(self.device)

        assignments = torch.zeros((bboxes.shape[0], )).to(self.device)
        cluster_centers = bboxes[cluster_center_idx]
        if self.num_anchors == 1:
            cluster_centers = self.kmeans_maximization(bboxes, assignments,
                                                       cluster_centers)
            anchors = bbox_xyxy_to_cxcywh(cluster_centers)[:, 2:].cpu().numpy()
            anchors = sorted(anchors, key=lambda x: x[0] * x[1])
            return anchors

        prog_bar = mmcv.ProgressBar(self.iters)
        for i in range(self.iters):
            converged, assignments = self.kmeans_expectation(
                bboxes, assignments, cluster_centers)
            if converged:
                self.logger.info(f'K-means process has converged at iter {i}.')
                break
            cluster_centers = self.kmeans_maximization(bboxes, assignments,
                                                       cluster_centers)
            prog_bar.update()
        print('\n')
        avg_iou = bbox_overlaps(bboxes,
                                cluster_centers).max(1)[0].mean().item()

        anchors = bbox_xyxy_to_cxcywh(cluster_centers)[:, 2:].cpu().numpy()
        anchors = sorted(anchors, key=lambda x: x[0] * x[1])
        self.logger.info(f'Anchor cluster finish. Average IOU: {avg_iou}')

        return anchors

    def kmeans_maximization(self, bboxes, assignments, centers):
        """Maximization part of EM algorithm(Expectation-Maximization)"""
        new_centers = torch.zeros_like(centers)
        for i in range(centers.shape[0]):
            mask = (assignments == i)
            if mask.sum():
                new_centers[i, :] = bboxes[mask].mean(0)
        return new_centers

    def kmeans_expectation(self, bboxes, assignments, centers):
        """Expectation part of EM algorithm(Expectation-Maximization)"""
        ious = bbox_overlaps(bboxes, centers)
        closest = ious.argmax(1)
        converged = (closest == assignments).all()
        return converged, closest


class YOLODEAnchorOptimizer(BaseAnchorOptimizer):
    """YOLO anchor optimizer using differential evolution algorithm.

    Args:
        num_anchors (int) : Number of anchors.
        iters (int): Maximum iterations for k-means.
        strategy (str): The differential evolution strategy to use.
            Should be one of:

                - 'best1bin'
                - 'best1exp'
                - 'rand1exp'
                - 'randtobest1exp'
                - 'currenttobest1exp'
                - 'best2exp'
                - 'rand2exp'
                - 'randtobest1bin'
                - 'currenttobest1bin'
                - 'best2bin'
                - 'rand2bin'
                - 'rand1bin'

            Default: 'best1bin'.
        population_size (int): Total population size of evolution algorithm.
            Default: 15.
        convergence_thr (float): Tolerance for convergence, the
            optimizing stops when ``np.std(pop) <= abs(convergence_thr)
            + convergence_thr * np.abs(np.mean(population_energies))``,
            respectively. Default: 0.0001.
        mutation (tuple[float]): Range of dithering randomly changes the
            mutation constant. Default: (0.5, 1).
        recombination (float): Recombination constant of crossover probability.
            Default: 0.7.
    """

    def __init__(self,
                 num_anchors,
                 iters,
                 strategy='best1bin',
                 population_size=15,
                 convergence_thr=0.0001,
                 mutation=(0.5, 1),
                 recombination=0.7,
                 **kwargs):

        super(YOLODEAnchorOptimizer, self).__init__(**kwargs)

        self.num_anchors = num_anchors
        self.iters = iters
        self.strategy = strategy
        self.population_size = population_size
        self.convergence_thr = convergence_thr
        self.mutation = mutation
        self.recombination = recombination

    def optimize(self):
        anchors = self.differential_evolution()
        self.save_result(anchors, self.out_dir)

    def differential_evolution(self):
        bboxes = self.get_zero_center_bbox_tensor()

        bounds = []
        for i in range(self.num_anchors):
            bounds.extend([(0, self.input_shape[0]), (0, self.input_shape[1])])

        result = differential_evolution(
            func=self.avg_iou_cost,
            bounds=bounds,
            args=(bboxes, ),
            strategy=self.strategy,
            maxiter=self.iters,
            popsize=self.population_size,
            tol=self.convergence_thr,
            mutation=self.mutation,
            recombination=self.recombination,
            updating='immediate',
            disp=True)
        self.logger.info(
            f'Anchor evolution finish. Average IOU: {1 - result.fun}')
        anchors = [(w, h) for w, h in zip(result.x[::2], result.x[1::2])]
        anchors = sorted(anchors, key=lambda x: x[0] * x[1])
        return anchors

    @staticmethod
    def avg_iou_cost(anchor_params, bboxes):
        assert len(anchor_params) % 2 == 0
        anchor_whs = torch.tensor(
            [[w, h]
             for w, h in zip(anchor_params[::2], anchor_params[1::2])]).to(
                 bboxes.device, dtype=bboxes.dtype)
        anchor_boxes = bbox_cxcywh_to_xyxy(
            torch.cat([torch.zeros_like(anchor_whs), anchor_whs], dim=1))
        ious = bbox_overlaps(bboxes, anchor_boxes)
        max_ious, _ = ious.max(1)
        cost = 1 - max_ious.mean().item()
        return cost


def main():
    logger = get_root_logger()
    args = parse_args()
    cfg = args.config
    cfg = Config.fromfile(cfg)

    input_shape = args.input_shape
    assert len(input_shape) == 2

    anchor_type = cfg.model.bbox_head.anchor_generator.type
    assert anchor_type == 'YOLOAnchorGenerator', \
        f'Only support optimize YOLOAnchor, but get {anchor_type}.'

    base_sizes = cfg.model.bbox_head.anchor_generator.base_sizes
    num_anchors = sum([len(sizes) for sizes in base_sizes])

    train_data_cfg = cfg.data.train
    while 'dataset' in train_data_cfg:
        train_data_cfg = train_data_cfg['dataset']
    dataset = build_dataset(train_data_cfg)

    if args.algorithm == 'k-means':
        optimizer = YOLOKMeansAnchorOptimizer(
            dataset=dataset,
            input_shape=input_shape,
            device=args.device,
            num_anchors=num_anchors,
            iters=args.iters,
            logger=logger,
            out_dir=args.output_dir)
    elif args.algorithm == 'differential_evolution':
        optimizer = YOLODEAnchorOptimizer(
            dataset=dataset,
            input_shape=input_shape,
            device=args.device,
            num_anchors=num_anchors,
            iters=args.iters,
            logger=logger,
            out_dir=args.output_dir)
    else:
        raise NotImplementedError(
            f'Only support k-means and differential_evolution, '
            f'but get {args.algorithm}')

    optimizer.optimize()


if __name__ == '__main__':
    main()
