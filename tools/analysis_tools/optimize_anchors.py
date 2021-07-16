import mmcv
import numpy as np
import torch
from mmcv import Config

from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.datasets.builder import build_dataset
from mmdet.utils import get_root_logger


class BaseAnchorOptimizer:

    def __init__(self, dataset, input_shape, logger):
        self.dataset = dataset
        self.input_shape = input_shape
        self.logger = logger
        bbox_whs, img_shapes = self.get_whs_and_shapes()
        ratios = img_shapes.max(1, keepdims=True) / np.array([input_shape])

        # resize to input shape
        self.bbox_whs = bbox_whs / ratios

    def get_whs_and_shapes(self):
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

    def optimize(self):
        raise NotImplementedError


def avg_iou_cost(x, bboxes):
    assert len(x) % 2 == 0
    anchor_whs = torch.tensor([[x[i], x[i + 1]]
                               for i in range(len(x) // 2)]).to(
                                   bboxes.device, dtype=bboxes.dtype)
    anchor_boxes = bbox_cxcywh_to_xyxy(
        torch.cat([torch.zeros_like(anchor_whs), anchor_whs], dim=1))
    ious = bbox_overlaps(bboxes, anchor_boxes)
    max_ious, _ = ious.max(1)
    cost = 1 - max_ious.mean().item()
    return cost


class YoloAnchorOptimizer(BaseAnchorOptimizer):

    def __init__(self,
                 dataset,
                 input_shape,
                 num_of_anchors,
                 kmeans_iters,
                 device='cuda',
                 **kwargs):

        super(YoloAnchorOptimizer, self).__init__(dataset, input_shape,
                                                  **kwargs)
        self.device = device
        self.num_of_anchors = num_of_anchors
        self.kmeans_iters = kmeans_iters

    def kmeans_anchors(self, num_centers, kmeans_iters):
        """https://github.com/AlexeyAB/darknet/blob/master/src/detector.c#L1421
        ."""
        self.logger.info(
            f'Start cluster {num_centers} YOLO anchors with K-means...')
        whs = torch.from_numpy(self.bbox_whs).to(
            self.device, dtype=torch.float32)
        bboxes = bbox_cxcywh_to_xyxy(
            torch.cat([torch.zeros_like(whs), whs], dim=1))
        cluster_center_idx = torch.randint(0, bboxes.shape[0],
                                           (num_centers, )).to(self.device)

        assignments = torch.zeros((bboxes.shape[0], )).to(self.device)
        cluster_centers = bboxes[cluster_center_idx]
        if num_centers == 1:
            cluster_centers = self.kmeans_maximization(bboxes, assignments,
                                                       cluster_centers)
            anchors = bbox_xyxy_to_cxcywh(cluster_centers)[:, 2:].cpu().numpy()
            anchors = sorted(anchors, key=lambda x: x[0] * x[1])
            return anchors

        prog_bar = mmcv.ProgressBar(kmeans_iters)
        for i in range(kmeans_iters):
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
        new_centers = torch.zeros_like(centers)
        for i in range(centers.shape[0]):
            mask = (assignments == i)
            if mask.sum():
                new_centers[i, :] = bboxes[mask].mean(0)
        return new_centers

    def kmeans_expectation(self, bboxes, assignments, centers):
        ious = bbox_overlaps(bboxes, centers)
        closest = ious.argmax(1)
        converged = (closest == assignments).all()
        return converged, closest

    def print_result(self, anchors):
        anchor_results = ''
        for w, h in anchors:
            anchor_results += f'({round(w)}, {round(h)}), '
        self.logger.info(f'Anchor cluster result:[{anchor_results}]')

    def optimize(self):
        anchors = self.kmeans_anchors(self.num_of_anchors, self.kmeans_iters)
        self.print_result(anchors)

    def differential_evolution(self):
        from scipy.optimize import differential_evolution

        whs = torch.from_numpy(self.bbox_whs).to(
            self.device, dtype=torch.float32)
        bboxes = bbox_cxcywh_to_xyxy(
            torch.cat([torch.zeros_like(whs), whs], dim=1))
        bounds = [(0, 104) for i in range(6)] + [
            (0, 208) for i in range(6)
        ] + [(0, 416) for i in range(6)]
        result = differential_evolution(
            avg_iou_cost,
            bounds=bounds,
            args=(bboxes, ),
            strategy='best1bin',
            maxiter=1000,
            updating='immediate',
            disp=True)
        print(result.x, result.fun)


if __name__ == '__main__':
    logger = get_root_logger()
    config_path = ''
    cfg = Config.fromfile(config_path)
    train_data_cfg = cfg.data.train
    while 'dataset' in train_data_cfg:
        train_data_cfg = train_data_cfg['dataset']
    dataset = build_dataset(train_data_cfg)
    opt = YoloAnchorOptimizer(dataset, [416, 416], 9, 1000, logger=logger)
    opt.differential_evolution()
