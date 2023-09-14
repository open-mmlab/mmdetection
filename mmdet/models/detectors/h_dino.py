# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType
from .dino import DINO


@MODELS.register_module()
class HDINO(DINO):

    def __init__(self,
                 *args,
                 bbox_head: OptConfigType = None,
                 **kwargs) -> None:
        self.num_query_one2one = bbox_head['num_query_one2one']
        super(HDINO, self).__init__(*args, bbox_head=bbox_head, **kwargs)

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[dict, dict]:
        decoder_inputs_dict, head_inputs_dict = super(HDINO, self).pre_decoder(
            memory, memory_mask, spatial_shapes, batch_data_samples)

        if self.training:
            dn_mask = decoder_inputs_dict['dn_mask']
            num_denoising_queries = head_inputs_dict['dn_meta'][
                'num_denoising_queries']
            num_query_one2one = num_denoising_queries + self.num_query_one2one
            dn_mask[num_query_one2one:, :num_query_one2one] = True
            dn_mask[num_denoising_queries:num_query_one2one,
                    num_query_one2one:] = True
            decoder_inputs_dict['dn_mask'] = dn_mask
        else:
            query = decoder_inputs_dict['query']
            reference_points = decoder_inputs_dict['reference_points']
            num_query_one2many = self.num_queries - self.num_query_one2one
            decoder_inputs_dict['query'] = query[:num_query_one2many]
            decoder_inputs_dict[
                'reference_points'] = reference_points[:num_query_one2many]
        return decoder_inputs_dict, head_inputs_dict
