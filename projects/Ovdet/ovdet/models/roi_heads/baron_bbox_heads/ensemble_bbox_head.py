# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn.functional as F
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.utils import empty_instances
from mmdet.registry import MODELS
from ....utils.misc import load_class_freq
from .convfc_bbox_head import BaronConvFCBBoxHead


@MODELS.register_module()
class EnsembleBaronConvFCBBoxHead(BaronConvFCBBoxHead):

    def __init__(self,
                 ensemble_factor=2.0 / 3.0,
                 class_info='data/metadata/lvis_v1_train_cat_norare_info.json',
                 kd=None,
                 transfer_factor=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.kd = MODELS.build(kd)
        self.ensemble_factor = ensemble_factor
        self.transfer_factor = transfer_factor
        assert (ensemble_factor is None) ^ (transfer_factor is None)
        if ensemble_factor is not None:
            class_cnt = load_class_freq(
                class_info, 1.0, min_count=0)  # to mask the novel classes
            is_base = torch.cat([(class_cnt > 0.0).float(),
                                 torch.tensor([1.0])])
            self.register_buffer('is_base', is_base)

    @staticmethod
    def _copy_params(src, dst):
        dst.load_state_dict(src.state_dict())

    def vision_to_language(self, x):
        return self.kd.vision_to_language(x)

    def forward(self, x, clip_model=None) -> tuple:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_score (Tensor): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_pred (Tensor): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """

        if not self.training:
            kd_pseudo_words = self.vision_to_language(x)
            kd_score = self.pred_cls_logits(kd_pseudo_words, clip_model)
            if self.loss_cls.use_sigmoid:
                kd_score = kd_score.sigmoid()
            else:
                kd_score = F.softmax(kd_score, dim=-1)

        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        pseudo_words = self.fc_cls(x_cls).view(-1, self.num_words,
                                               self.word_dim)
        cls_score = self.pred_cls_logits(pseudo_words, clip_model)
        if not self.training:
            if self.loss_cls.use_sigmoid:
                cls_score = cls_score.sigmoid()
            else:
                cls_score = F.softmax(cls_score, dim=-1)
            if self.ensemble_factor is not None:
                assert self.ensemble_factor > 0.5
                base_score = (cls_score ** self.ensemble_factor) * \
                             (kd_score ** (1.0 - self.ensemble_factor)) * \
                    self.is_base[None]
                novel_score = (cls_score ** (1.0 - self.ensemble_factor)) * \
                              (kd_score ** self.ensemble_factor) * \
                    (1.0 - self.is_base[None])
                cls_score = base_score + novel_score
            else:
                assert self.transfer_factor is not None
                cls_score = (cls_score ** self.transfer_factor) * \
                            (kd_score ** (1.0 - self.transfer_factor))

        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred

    def _predict_by_feat_single(
            self,
            roi: Tensor,
            cls_score: Tensor,
            bbox_pred: Tensor,
            img_meta: dict,
            rescale: bool = False,
            rcnn_test_cfg: Optional[ConfigDict] = None) -> InstanceData:
        if roi.shape[0] == 0:
            results = InstanceData()
            return empty_instances([img_meta],
                                   roi.device,
                                   task_type='bbox',
                                   instance_results=[results],
                                   box_type=self.predict_box_type,
                                   use_box_type=False,
                                   num_classes=self.num_classes,
                                   score_per_cls=rcnn_test_cfg is None)[0]

        return self._predict_after_normalize_cls_score(roi, cls_score,
                                                       bbox_pred, img_meta,
                                                       rescale, rcnn_test_cfg)


@MODELS.register_module()
class EnsembleBaronShared2FCBBoxHead(EnsembleBaronConvFCBBoxHead):

    def __init__(self, fc_out_channels: int = 1024, *args, **kwargs) -> None:
        super().__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@MODELS.register_module()
class EnsembleBaronShared4Conv1FCBBoxHead(EnsembleBaronConvFCBBoxHead):

    def __init__(self, fc_out_channels: int = 1024, *args, **kwargs) -> None:
        super().__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
