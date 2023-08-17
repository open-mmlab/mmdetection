import torch.nn as nn
from mmengine.model import BaseModule
from mmengine.model.weight_init import constant_init
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
# from mmpl.registry import MODELS
from mmdet.registry import MODELS
from mmengine.model import BaseModule
from transformers import GPT2Model, GPT2Config


@MODELS.register_module()
class HFGPTTransformerDecoderNeck(BaseModule):
    def __init__(
            self,
            model_name='gpt2',
            from_pretrained=True,
            update_kwargs=dict(
                max_position_embeddings=512,
                hidden_size=512,
            )
    ):
        super(HFGPTTransformerDecoderNeck, self).__init__()
        self.model_name = model_name
        if from_pretrained:
            self.gpt_model = GPT2Model.from_pretrained(model_name)
        else:
            config = GPT2Config.from_pretrained(model_name)
            config.update(update_kwargs)
            self.gpt_model = GPT2Model(config=config)
            # self.wte = nn.Embedding(0, 512)

    def forward(self, *args, **kwargs):
        out_puts = self.gpt_model(*args, **kwargs)
        return out_puts
