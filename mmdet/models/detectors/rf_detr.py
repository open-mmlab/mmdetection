# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import Any

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh

from ..layers.rf_detr.config import RFDETRSmallConfig, TrainConfig
from ..layers.rf_detr.models.lwdetr import (build_criterion_from_config,
                                            build_model_from_config)
from ..layers.rf_detr.utilities.tensors import NestedTensor
from .base import BaseDetector


@MODELS.register_module()
class RFDETR(BaseDetector):
    """Native MMDetection wrapper for RF-DETR detection models.

    The wrapped network is the vendored RF-DETR LWDETR module.  This class owns
    MMDetection/MMEngine integration: ``DetDataSample`` conversion, loss and
    predict modes, and ``InstanceData`` outputs.
    """

    config_cls = RFDETRSmallConfig

    def __init__(
        self,
        num_classes: int = 80,
        model_config: dict[str, Any] | None = None,
        train_config: dict[str, Any] | None = None,
        score_thr: float = 0.0,
        data_preprocessor: dict | None = None,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        model_cfg = dict(model_config or {})
        model_cfg.setdefault('num_classes', num_classes)
        # MMEngine checkpoint loading owns pretrained weights.  Avoid implicit
        # RF-DETR cache downloads during model construction.
        model_cfg.setdefault('pretrain_weights', None)
        model_cfg.setdefault('device', 'cpu')
        self.model_config = self.config_cls(**model_cfg)

        train_cfg = dict(train_config or {})
        train_cfg.setdefault('dataset_dir', '.')
        train_cfg.setdefault('output_dir', 'work_dirs/rf_detr')
        self.train_config = TrainConfig(**train_cfg)

        self.model = build_model_from_config(self.model_config, self.train_config)
        self.criterion, self.postprocessor = build_criterion_from_config(
            self.model_config, self.train_config)
        self.score_thr = score_thr
        self.num_classes = num_classes

    def extract_feat(self, batch_inputs: Tensor) -> tuple[Tensor]:
        """Return the raw input tensor for MMDetection compatibility.

        RF-DETR's DINOv2 backbone consumes ``NestedTensor`` objects internally,
        so feature extraction is not exposed as separate FPN tensors.
        """
        return (batch_inputs, )

    def _make_nested_tensor(self, batch_inputs: Tensor,
                            batch_data_samples: SampleList | None) -> NestedTensor:
        if batch_data_samples is None:
            mask = torch.zeros(
                batch_inputs.shape[0],
                batch_inputs.shape[-2],
                batch_inputs.shape[-1],
                dtype=torch.bool,
                device=batch_inputs.device)
            return NestedTensor(batch_inputs, mask)

        masks = []
        pad_h, pad_w = batch_inputs.shape[-2:]
        for data_sample in batch_data_samples:
            img_h, img_w = data_sample.metainfo.get(
                'batch_input_shape',
                data_sample.metainfo.get('img_shape',
                                         (pad_h, pad_w)))[:2]
            mask = torch.ones(
                (pad_h, pad_w), dtype=torch.bool, device=batch_inputs.device)
            mask[:img_h, :img_w] = False
            masks.append(mask)
        return NestedTensor(batch_inputs, torch.stack(masks, dim=0))

    def _targets_from_samples(self,
                              batch_data_samples: SampleList) -> list[dict[str,
                                                                           Tensor]]:
        targets: list[dict[str, Tensor]] = []
        for data_sample in batch_data_samples:
            gt_instances = data_sample.gt_instances
            device = gt_instances.bboxes.device
            img_h, img_w = data_sample.metainfo.get(
                'img_shape',
                data_sample.metainfo.get('batch_input_shape'))[:2]
            bboxes = gt_instances.bboxes.tensor if hasattr(
                gt_instances.bboxes, 'tensor') else gt_instances.bboxes
            bboxes = bboxes.to(device=device, dtype=torch.float32)
            if bboxes.numel() == 0:
                boxes = bboxes.new_zeros((0, 4))
            else:
                scale = bboxes.new_tensor([img_w, img_h, img_w, img_h])
                boxes = bbox_xyxy_to_cxcywh(bboxes) / scale
                boxes = boxes.clamp(min=0.0, max=1.0)
            labels = gt_instances.labels.to(device=device, dtype=torch.long)
            targets.append({'boxes': boxes, 'labels': labels})
        return targets

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict[str, Tensor]:
        nested = self._make_nested_tensor(batch_inputs, batch_data_samples)
        targets = self._targets_from_samples(batch_data_samples)
        outputs = self.model(nested, targets)
        raw_losses = self.criterion(outputs, targets)
        return {
            name: loss * self.criterion.weight_dict.get(name, 1.0)
            for name, loss in raw_losses.items()
            if name in self.criterion.weight_dict
        }

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        nested = self._make_nested_tensor(batch_inputs, batch_data_samples)
        outputs = self.model(nested)
        target_sizes = []
        for data_sample in batch_data_samples:
            ori_shape = data_sample.metainfo.get(
                'ori_shape', data_sample.metainfo.get('img_shape'))
            target_sizes.append(
                batch_inputs.new_tensor(ori_shape[:2], dtype=torch.float32))
        results = self.postprocessor(outputs, torch.stack(target_sizes, dim=0))
        pred_instances = []
        for result in results:
            keep = (result['scores'] >= self.score_thr) & (
                result['labels'] < self.num_classes)
            pred_instances.append(
                InstanceData(
                    bboxes=result['boxes'][keep],
                    scores=result['scores'][keep],
                    labels=result['labels'][keep]))
        return self.add_pred_to_datasample(batch_data_samples, pred_instances)

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: SampleList | None = None) -> dict[str, Any]:
        nested = self._make_nested_tensor(batch_inputs, batch_data_samples)
        return self.model(nested)
