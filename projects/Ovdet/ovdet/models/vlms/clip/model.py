from mmengine.model import BaseModule

from mmdet.registry import MODELS


@MODELS.register_module()
class CLIP(BaseModule):

    def __init__(self, text_encoder, image_encoder):
        super().__init__()
        if text_encoder is not None:
            self.text_encoder = MODELS.build(text_encoder)
        if image_encoder is not None:
            self.image_encoder = MODELS.build(image_encoder)
