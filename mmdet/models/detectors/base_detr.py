# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Union

from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector


@MODELS.register_module()
class TransformerDetector(BaseDetector, metaclass=ABCMeta):
    """Base class for Transformer-based detectors.

    Transformer-based detectors use an encoder to process output features of
    neck, then several queries interactive with the output features of encoder
    and do the regression and classification.

    Args:
        backbone (:obj:`ConfigDict` or dict): Config of backbone.
        neck (:obj:`ConfigDict` or dict): Config of neck.
            Defaults to None.
        encoder_cfg (:obj:`ConfigDict` or dict): Config of encoder.
            Defaults to None.
        decoder_cfg (:obj:`ConfigDict` or dict): Config of decoder.
            Defaults to None.
        positional_encoding_cfg (:obj:`ConfigDict` or dict): Config of
            positional encoding. Defaults to None.
        bbox_head (:obj:`ConfigDict` or dict): Config for position
            encoding. Defaults to None.
        num_query (int): Number of decoder query in Transformer.
            Defaults to 100.
        train_cfg (:obj:`ConfigDict` or dict): Training config of transformer
            head. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict): Testing config of transformer
            head. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
           config of :class:`BaseDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 encoder_cfg: OptConfigType = None,
                 decoder_cfg: OptConfigType = None,
                 positional_encoding_cfg: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 num_query: int = 100,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        # process args
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.encoder_cfg = encoder_cfg
        self.decoder_cfg = decoder_cfg
        self.positional_encoding_cfg = positional_encoding_cfg
        self.num_query = num_query

        # init model layers
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.bbox_head = MODELS.build(bbox_head)
        self._init_layers()

    @abstractmethod
    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        pass

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[Tensor]: A tuple of features from ``bbox_head`` forward.
        """
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        results = self.bbox_head.forward(
            **head_inputs_dict)  # TODO: refine this  # noqa
        return results

    def forward_transformer(self, img_feats,
                            batch_data_samples) -> Dict:  # TODO: typehint
        # TODO: Doc
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        memory = self.forward_encoder(**encoder_inputs_dict)

        temp_dec_in, head_inputs_dict = self.pre_decoder(memory)
        decoder_inputs_dict.update(temp_dec_in)  # TODO: refine 'update'

        temp_head_in = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(temp_head_in)
        return head_inputs_dict

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
                different resolutions.
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    @abstractmethod
    def pre_transformer(
        self,
        img_feats: Tuple[Tensor],
        batch_data_samples: OptSampleList = None
    ) -> Tuple[Dict, Dict]:  # TODO: typehint
        """This function creates the inputs of the Transformer.

        1. Construct batch padding mask.
        2. Prepare transformer_inputs_dict.
        """
        pass

    @abstractmethod
    def forward_encoder(self, **kwargs) -> Tensor:  # TODO: typehint
        """TODO: Doc"""
        pass

    @abstractmethod
    def pre_decoder(self, **kwargs) -> Tuple[Dict, Dict]:  # TODO: typehint
        """TODO: Doc"""
        pass

    @abstractmethod
    def forward_decoder(self, **kwargs) -> Dict:  # TODO: typehint
        """TODO: Doc"""
        pass
