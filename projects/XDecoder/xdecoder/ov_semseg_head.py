from torch import nn


class XDecoderOVSemSegHead(nn.Module):
    def __init__(self, input_shape,
                 num_classes: int,
                 pixel_decoder: nn.Module,
                 transformer_decoder: nn.Module,
                 transformer_in_feature: str,
                 ignore_value: int = -1):
        super().__init__()

    def predict(self, features):
        mask_features, multi_scale_features = self.pixel_decoder(features)
        predictions = self.predictor(multi_scale_features, mask_features)
        return predictions

