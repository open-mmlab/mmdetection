# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import Linear
from mmengine.model import bias_init_with_prob, constant_init
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import InstanceList, OptInstanceList
from ..layers import inverse_sigmoid
from .detr_head import DETRHead


@MODELS.register_module()
class DeformableDETRHead(DETRHead):
    """Head of DeformDETR: Deformable DETR: Deformable Transformers for End-to-
    End Object Detection.

    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.
    """

    def __init__(self,
                 *args,
                 as_two_stage: bool = False,
                 with_box_refine: bool = False,
                 num_decoder_layers: int = 6,
                 **kwargs) -> None:
        # NOTE The three key word args are set in the detector,
        # the users do not need to set them in config.
        self.as_two_stage = as_two_stage
        self.with_box_refine = with_box_refine
        self.num_decoder_layers = num_decoder_layers

        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.num_decoder_layers + 1) if \
            self.as_two_stage else self.num_decoder_layers

        if self.with_box_refine:
            self.cls_branches = nn.ModuleList(
                [copy.deepcopy(fc_cls) for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [copy.deepcopy(reg_branch) for _ in range(num_pred)])
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

    def init_weights(self) -> None:
        """Initialize weights of the DeformDETR head."""
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def forward(self, hidden_states: Tensor,
                references: List[Tensor]) -> Tuple[Tensor]:
        """Forward function.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape
                [num_decoder_layers, num_query, bs, embed_dims].
            references (List[Tensor]): List of the reference of the decoder.
                The first reference is the init_reference and the other
                num_decoder_layers(6) references are inter_reference.
                The init_reference has shape [bs, num_query, 4] when
                `as_two_stage` is True, otherwise [bs, num_query, 2].
                Each inter_reference has shape [bs, num_query, 4] when
                `with_box_refine` is True, otherwise [bs, num_query, 2].

        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

            - outputs_classes (Tensor): Outputs from the classification head,
              shape [nb_dec, bs, num_query, cls_out_channels].
            - outputs_coords (Tensor): Sigmoid outputs from the regression
              head with normalized coordinate format (cx, cy, w, h).
              Shape [nb_dec, bs, num_query, 4].
        """
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []

        for layer in range(hidden_states.shape[0]):  # TODO
            reference = inverse_sigmoid(references[layer])
            # NOTE The last reference will not be used.
            outputs_class = self.cls_branches[layer](hidden_states[layer])
            tmp = self.reg_branches[layer](hidden_states[layer])
            if reference.shape[-1] == 4:
                # TODO: comment
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        return outputs_classes, outputs_coords

    def loss(self, hidden_states: Tensor, references: List[Tensor],
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape
                [num_decoder_layers, num_query, bs, embed_dims].
            references (List[Tensor]): List of the reference of the decoder.
                The first reference is the init_reference and the other
                num_decoder_layers(6) references are inter_reference.
                The init_reference has shape [bs, num_query, 4] when
                `as_two_stage` is True, otherwise [bs, num_query, 2].
                Each inter_reference has shape [bs, num_query, 4] when
                `with_box_refine` is True, otherwise [bs, num_query, 2].
            enc_outputs_class (Tensor): The score of each point on encode
                feature map, has shape (N, num_features, cls_out_channels).
                Only when as_two_stage is True it would be returned,
                otherwise `None` would be returned.
            enc_outputs_coord (Tensor): The proposal generate from the
                encode feature map, has shape (N, num_features, 4). Only when
                as_two_stage is True it would be returned, otherwise `None`
                would be returned.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references)
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              batch_gt_instances, batch_img_metas)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(
        self,
        all_cls_scores: Tensor,
        all_bbox_preds: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """"Loss function.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, batch_size, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, batch_size, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        loss_dict = super().loss_by_feat(all_cls_scores, all_bbox_preds,
                                         batch_gt_instances, batch_img_metas,
                                         batch_gt_instances_ignore)

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            proposal_gt_instances = copy.deepcopy(batch_gt_instances)
            for i in range(len(proposal_gt_instances)):
                proposal_gt_instances[i].labels = torch.zeros_like(
                    proposal_gt_instances[i].labels)
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_by_feat_single(
                    enc_cls_scores, enc_bbox_preds,
                    batch_gt_instances=proposal_gt_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou
        return loss_dict

    def predict(self,
                hidden_states: Tensor,
                references: List[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = True,
                **kwargs) -> InstanceList:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape
                [num_decoder_layers, num_query, bs, embed_dims].
            references (List[Tensor]): List of the reference of the decoder.
                The first reference is the init_reference and the other
                num_decoder_layers(6) references are inter_reference.
                The init_reference has shape [bs, num_query, 4] when
                `as_two_stage` is True, otherwise [bs, num_query, 2].
                Each inter_reference has shape [bs, num_query, 4] when
                `with_box_refine` is True, otherwise [bs, num_query, 2].
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): If True, return boxes in original
                image space. Defaults to True.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        outs = self(hidden_states, references)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions

    def predict_by_feat(self,
                        all_cls_scores: Tensor,
                        all_bbox_preds: Tensor,
                        batch_img_metas: List[Dict],
                        rescale: bool = False) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, batch_size, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, batch_size, num_query, 4].
            batch_img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Default False.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]

        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(cls_score, bbox_pred,
                                                   img_meta, rescale)
            result_list.append(results)
        return result_list
