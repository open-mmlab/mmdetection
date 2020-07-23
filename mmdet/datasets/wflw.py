# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Sebastien Martin(sebastienmartin@linkernetworks.com)
# ------------------------------------------------------------------------------

import copy
import os
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..core.keypoint import compute_nme
from .builder import DATASETS, PIPELINES
from .custom import CustomDataset


@PIPELINES.register_module()
class FocusOnFace(object):
    """After loading image, we only want a sub part of that said image."""

    def __init__(self, out_size=384):
        """[summary]

        Args:
            out_size (int, optional): Specifies the input size of focused
            before pre-processing (i.e. before resizing). Defaults to 384.
        """
        if not isinstance(out_size, Sequence):
            out_size = [out_size, out_size]
        self.out_size = np.array(out_size)

    def __call__(self, results):
        img_info = copy.deepcopy(results['img_info'])
        ann = img_info['ann']
        # Shift annotations
        # NOTE: Assumes only one bbox or keypoints set per image (same
        # as in original HRNet code)
        bbox = ann['bboxes'][0]
        pts = ann['keypoints'][0]
        shift = ((bbox[2:] + bbox[:2]) / 2 - (self.out_size / 2)).astype(
            np.int)
        shift[shift < 0] = 0  # Already on the left border
        shifted_pts = pts[:, :2] - shift
        invisible = np.logical_or(
            np.any(shifted_pts[:, :2] < 0, axis=1),
            np.logical_or(shifted_pts[:, 0] > self.out_size[0],
                          shifted_pts[:, 1] > self.out_size[1]))
        shifted_pts = np.hstack(
            (shifted_pts, pts[:, 2].reshape(-1, 1)))  # Add visibility
        if np.any(invisible):
            shifted_pts[:,
                        2][invisible] = np.fmin(shifted_pts[:, 2][invisible],
                                                0)
        im_pts = np.zeros((1, shifted_pts.shape[0], 3), dtype=np.float32)
        im_pts[0, :, :] = shifted_pts
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        new_box = bbox - np.tile(shift, 2)
        bboxes = np.array(new_box, dtype=np.float32).reshape((-1, 4))  # bbox
        img_info.update({'width': w, 'height': h, 'shift': shift})
        ann.update({
            'bboxes': bboxes,
            'keypoints': im_pts,
            'n_kpts': im_pts.shape[1]
        })
        results['img_info'] = img_info
        results['ann_info'] = ann
        # Extract image ROI
        if 'img' in results:
            img = results['img']
            results['img'] = img[shift[1]:shift[1] + self.out_size[1],
                                 shift[0]:shift[0] + self.out_size[0], :]
            results['img_shape'] = results['img'].shape
            results['ori_shape'] = results['img'].shape
            # Set initial values for default meta_keys
            results['pad_shape'] = results['img'].shape
        results['scale'] = tuple(self.out_size)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}()')
        return repr_str


@DATASETS.register_module()
class WflwDataset(CustomDataset):

    def __init__(self, **kwargs):
        super(WflwDataset, self).__init__(**kwargs)
        # specify annotation file for dataset
        self.is_train = not self.test_mode
        if self.test_mode:
            # Faster RCNN complains otherwise
            self.proposals = [0 for i in range(len(self))]
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def load_annotations(self, ann_file):
        # Need width, height, keypoints, filename, imageID,
        # Keypoints are represented as an N*K*3 array, for N persons, and K
        # keypoints per person
        # For each keypoint, we record X and Y position, as well as visibility
        # flag (see COCO dataset)
        anns = pd.read_csv(ann_file, sep=' ', header=None)
        out_anns = []
        n_kpts = 98
        for idx in range(anns.shape[0]):
            bboxes = np.zeros((1, 4), dtype=np.float32)
            labels = np.zeros((1), dtype=np.int64)
            pts = anns.iloc[idx, :(n_kpts * 2)].values
            bbox = anns.iloc[idx, 196:200].values.astype('float32').reshape(4)
            pts = pts.astype('float32').reshape(n_kpts, 2)

            bboxes[0, :] = bbox
            im_pts = np.ones((1, pts.shape[0], 3), dtype=np.float32) * 2
            im_pts[0, :, :2] = pts
            im_name = anns.iloc[idx, -1]
            im_name = os.path.join(self.img_prefix, im_name)
            w, h = 50, 50  # Just make it > 32 so it won't be discarded later,
            # need to focus on face later
            ann = {
                'width': w,
                'height': h,
                'filename': im_name,
                'image_id': idx,
                'ann': {
                    'bboxes': bboxes,
                    'keypoints': im_pts,
                    'labels': labels,
                    'n_kpts': im_pts.shape[1]
                }
            }
            out_anns.append(ann)
        return out_anns

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 kpts_thr=0.1,
                 scale_ranges=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            kpts_thr (float): Keypoints confidence threshold
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        if metric not in ('keypoint', ):
            res_bboxes = [
                res[0] if isinstance(res, tuple) else res for res in results
            ]
            eval_results = super().evaluate(res_bboxes, metric, logger,
                                            proposal_nums, iou_thr,
                                            scale_ranges)
        else:
            # Evaluate keypoints
            res_kpts = [res['keypoints'] for res in results]
            data_infos = [self.data_infos[i] for i in range(len(self))]
            nme_count = 0
            nme_batch_sum = 0
            for res_kpt, gt in zip(res_kpts, data_infos):
                gt = FocusOnFace()({'img_info': gt})['img_info']
                pred = res_kpt
                gt_keypoints = gt['ann']['keypoints']
                nme_batch = compute_nme(pred, gt_keypoints, normalize=True)
                nme_batch_sum = nme_batch_sum + np.sum(nme_batch)
                nme_count = nme_count + pred.shape[0]
            eval_results = {'keypoints': nme_batch_sum / nme_count}
        return eval_results


if __name__ == '__main__':
    pass
