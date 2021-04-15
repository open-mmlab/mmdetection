from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16
from mmcv.utils import print_log

from mmdet.models.builder import HEADS
from .semantic_head import SemanticHead


@HEADS.register_module()
class FusedSemanticHead(SemanticHead):
    r"""Semantic Segmentation Head used in Hybrid Task Cascade (HTC).

    Based on the original semantic segmentation head, this head further
    outputs a semantic feature map for RoI pooling to assist the
    instance mask and bbox heads.

    Args:
        conv_out_channels (int, optional): Output channels of feature.
            Defaults to 256.

    """
    _version = 2

    def __init__(self, *args, conv_out_channels=256, **kwargs):
        super(FusedSemanticHead, self).__init__(*args, **kwargs)
        self.conv_out_channels = conv_out_channels
        self.conv_embedding = ConvModule(
            conv_out_channels,
            conv_out_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if version is None or version < 2:
            # the key is different in early versions
            # In version < 2, FusedSemanticHead loads old models.
            # The conv layers have been moved into self.semantic_decoder
            for i in range(5):
                for conv_name in ['lateral_convs', 'convs']:
                    # conv weight
                    new_name = f'{prefix}semantic_decoder.' + \
                        f'{conv_name}.{i}.conv.weight'
                    old_name = f'{prefix}{conv_name}.{i}.conv.weight'
                    if (new_name not in state_dict and old_name in state_dict):
                        state_dict[new_name] = state_dict.pop(old_name)

                    # conv bias
                    new_name = f'{prefix}semantic_decoder.{conv_name}' + \
                        '.{i}.conv.bias'
                    old_name = f'{prefix}{conv_name}.{i}.conv.bias'
                    if (new_name not in state_dict and old_name in state_dict):
                        state_dict[new_name] = state_dict.pop(old_name)

        if version is not None and version > 1:
            print_log(
                f'FusedSemanticHead {prefix.rstrip(".")} is upgraded to '
                'version 2.',
                logger='root')

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    @auto_fp16()
    def forward(self, feats):
        x = self.semantic_decoder(feats)
        mask_pred = self.conv_logits(x)
        x = self.conv_embedding(x)
        return mask_pred, x
