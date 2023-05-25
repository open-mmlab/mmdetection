from torch import nn
from mmdet.registry import MODELS
import copy


@MODELS.register_module()
class XDecoderOVSemSegHead(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 pixel_decoder: nn.Module,
                 transformer_decoder: nn.Module,
                 ignore_value: int = 255,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super().__init__()
        self.ignore_value = ignore_value
        self.common_stride = 4
        self.num_classes = num_classes

        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(
            in_channels=in_channels)
        self.pixel_decoder = MODELS.build(pixel_decoder_)
        self.predictor = MODELS.build(transformer_decoder)

    def predict(self, features, batch_data_samples, rescale=True):
        text_prompts = [
            data_samples.caption for data_samples in batch_data_samples
        ]
        self.predictor.lang_encoder.get_text_embeddings(text_prompts + ["background"], is_eval=True)
        mask_features, multi_scale_features = self.pixel_decoder(features)
        predictions = self.predictor(multi_scale_features, mask_features)

        return predictions
