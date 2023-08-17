# import torch
# from mmengine.structures import InstanceData
# from typing import List, Any

# from mmdet.registry import MODELS
# from mmseg.utils import SampleList
# from .base_pler import BasePLer
# import torch.nn.functional as F
# from ..sam import sam_model_registry


# @MODELS.register_module()
# class SegSAMAnchor(BasePLer):
#     def __init__(self,
#                  backbone,
#                  neck=None,
#                  panoptic_head=None,
#                  need_train_names=None,
#                  train_cfg=None,
#                  test_cfg=None,
#                  *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.save_hyperparameters()
#         self.need_train_names = need_train_names

#         backbone_type = backbone.pop('type')
#         self.backbone = sam_model_registry[backbone_type](**backbone)

#         if neck is not None:
#             self.neck = MODELS.build(neck)

#         self.panoptic_head = MODELS.build(panoptic_head)

#         self.train_cfg = train_cfg
#         self.test_cfg = test_cfg

#     def setup(self, stage: str) -> None:
#         super().setup(stage)
#         if self.need_train_names is not None:
#             self._set_grad(self.need_train_names, noneed_train_names=[])

#     def init_weights(self):
#         import ipdb; ipdb.set_trace()
#         pass

#     def train(self, mode=True):
#         if self.need_train_names is not None:
#             return self._set_train_module(mode, self.need_train_names)
#         else:
#             super().train(mode)
#             return self

#     @torch.no_grad()
#     def extract_feat(self, batch_inputs):
#         feat, inter_features = self.backbone.image_encoder(batch_inputs)
#         return feat, inter_features

#     def validation_step(self, batch, batch_idx):
#         data = self.data_preprocessor(batch, False)
#         batch_inputs = data['inputs']
#         batch_data_samples = data['data_samples']

#         x = self.extract_feat(batch_inputs)
#         # x = (
#         # torch.rand(2, 256, 64, 64).to(self.device), [torch.rand(2, 64, 64, 768).to(self.device) for _ in range(12)])
#         results = self.panoptic_head.predict(
#             x, batch_data_samples, self.backbone)
#         self.val_evaluator.update(batch, results)

#     def training_step(self, batch, batch_idx):
#         data = self.data_preprocessor(batch, True)
#         batch_inputs = data['inputs']
#         batch_data_samples = data['data_samples']
#         x = self.extract_feat(batch_inputs)
#         # x = (torch.rand(2, 256, 64, 64).to(self.device), [torch.rand(2, 64, 64, 768).to(self.device) for _ in range(12)])
#         losses = self.panoptic_head.loss(x, batch_data_samples, self.backbone)

#         parsed_losses, log_vars = self.parse_losses(losses)
#         log_vars = {f'train_{k}': v for k, v in log_vars.items()}
#         log_vars['loss'] = parsed_losses
#         self.log_dict(log_vars, prog_bar=True)
#         return log_vars

#     def on_before_optimizer_step(self, optimizer) -> None:
#         self.log_grad(module=self.panoptic_head)

#     def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
#         data = self.data_preprocessor(batch, False)
#         batch_inputs = data['inputs']
#         batch_data_samples = data['data_samples']

#         x = self.extract_feat(batch_inputs)
#         # x = (
#         # torch.rand(2, 256, 64, 64).to(self.device), [torch.rand(2, 64, 64, 768).to(self.device) for _ in range(12)])
#         results = self.panoptic_head.predict(
#             x, batch_data_samples, self.backbone)
#         return results

#     def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
#         data = self.data_preprocessor(batch, False)
#         batch_inputs = data['inputs']
#         batch_data_samples = data['data_samples']

#         x = self.extract_feat(batch_inputs)
#         # x = (
#         # torch.rand(2, 256, 64, 64).to(self.device), [torch.rand(2, 64, 64, 768).to(self.device) for _ in range(12)])
#         results = self.panoptic_head.predict(
#             x, batch_data_samples, self.backbone)
#         self.test_evaluator.update(batch, results)





import torch
# from mmengine.structures import InstanceData
from typing import List, Any
from mmengine.model.base_model import BaseModel

# from mmseg.utils import SampleList
# import torch.nn.functional as F
from ..sam import sam_model_registry

# from rssam.registry import MODELS
from mmdet.registry import MODELS

@MODELS.register_module()
class SegSAMAnchor(BaseModel):
    def __init__(self,
                 backbone,
                 neck=None,
                 panoptic_head=None,
                 need_train_names=None,
                 train_cfg=None,
                 test_cfg=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.need_train_names = need_train_names

        backbone_type = backbone.pop('type')
        self.backbone = sam_model_registry[backbone_type](**backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)

        self.panoptic_head = MODELS.build(panoptic_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def init_weights(self):
        pass

    @torch.no_grad()
    def extract_feat(self, batch_inputs):
        feat, inter_features = self.backbone.image_encoder(batch_inputs)
        return feat, inter_features

    def forward(self, inputs, data_samples, mode='tensor'):
        if mode == 'loss':
            x = self.extract_feat(inputs)
            losses = self.panoptic_head.loss(x, data_samples, self.backbone)
            return losses
        elif mode == 'predict':
            x = self.extract_feat(inputs)
            results = self.panoptic_head.predict(x, data_samples, self.backbone)
            return results
        elif mode == 'tensor':
            x = self.extract_feat(inputs)
            return x
