import imp
from multiprocessing.spawn import import_main_path

import torch
import torch.nn as nn
from mmcv.ops import batched_nms
from mmengine.data import InstanceData


from mmdet.structures import DetDataSample
from mmdet.structures.bbox import  bbox_flip
from mmdet.models.builder import MODELS

from mmengine.model import TestTimeAugModelWrapper


@MODELS.register_module()
class TestAugModelWrapper(TestTimeAugModelWrapper):
    def __init__(self, module):
        super(TestAugModelWrapper, self).__init__()
        self.module = MODELS.build(module)

    def collate_data(self, data_batch):
        first_inputs = data_batch[0]['inputs']
        num_augs = len(first_inputs)
        num_imgs = len(data_batch)
        output = []
        for aug_idx in range(num_augs):
            _single_batch = []
            for batch_idx in range(num_imgs):
                _data = dict()
                for key in data_batch[0].keys():
                    _data[key] = data_batch[batch_idx][key][aug_idx]
                _single_batch.append(_data)
            output.append(_single_batch)
        for data in data_batch:
            data['data_sample'] = data['data_sample'][0]
        return output

    def merge_results(self, data_list, result_list):
        aug_bboxes = []
        aug_scores = []
        aug_labels = []
        data_list = self.collate_data(data_list)
        img_metas = []
        for data, predictions in zip(data_list, result_list):
            _img_metas = []
            aug_bboxes.append(predictions[0].pred_instances.bboxes)
            aug_scores.append(predictions[0].pred_instances.scores)
            aug_labels.append(predictions[0].pred_instances.labels)
            _img_metas.append(data[0]['data_sample'].metainfo)
            img_metas.append(_img_metas)

        merged_bboxes, merged_scores = self.merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas)
        merged_labels = torch.cat(aug_labels, dim=0) if aug_labels else None

        if merged_bboxes.numel() == 0:
            det_bboxes = torch.cat([merged_bboxes, merged_scores[:, None]], -1)
            return [
                (det_bboxes, merged_labels),
            ]

        if hasattr(self.module, 'module'):
            model = self.module.module
        else:
            model = self.module
        det_bboxes, keep_idxs = batched_nms(
            merged_bboxes,
            merged_scores,
            merged_labels,
            model.bbox_head.test_cfg.nms)
        det_bboxes = det_bboxes[:model.bbox_head.test_cfg.max_per_img]
        det_labels = merged_labels[keep_idxs][:model.bbox_head.test_cfg.
                                              max_per_img]

        results = InstanceData()

        _det_bboxes = det_bboxes.clone()

        results.bboxes = _det_bboxes[:, :4]
        results.scores = _det_bboxes[:, 4]
        results.labels = det_labels
        det_results = DetDataSample()
        det_results.pred_instances = results
        return [det_results]

    def merge_aug_bboxes(self, aug_bboxes, aug_scores, img_metas):
        """Merge augmented detection bboxes and scores.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 4*#class)
            aug_scores (list[Tensor] or None): shape (n, #class)
            img_shapes (list[Tensor]): shape (3, ).

        Returns:
            tuple[Tensor]: ``bboxes`` with shape (n,4), where
            4 represent (tl_x, tl_y, br_x, br_y)
            and ``scores`` with shape (n,).
        """
        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            ori_shape = img_info[0]['ori_shape']
            flip = img_info[0]['flip']
            flip_direction = img_info[0]['flip_direction']
            if flip:
                bboxes = bbox_flip(bboxes=bboxes, img_shape=ori_shape, direction=flip_direction)
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        if aug_scores is None:
            return bboxes
        else:
            scores = torch.cat(aug_scores, dim=0)
            return bboxes, scores


