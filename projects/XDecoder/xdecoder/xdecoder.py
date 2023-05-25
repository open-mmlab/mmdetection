from torch import Tensor
import torch
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.single_stage import SingleStageDetector
from torch.nn import functional as F
from .utils import retry_if_cuda_oom, sem_seg_postprocess
from mmengine.structures import PixelData


@MODELS.register_module()
class XDecoder(SingleStageDetector):
    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 semseg_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super(SingleStageDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)

        self.sem_seg_head = None
        if semseg_head is not None:
            semseg_head_ = semseg_head.deepcopy()
            semseg_head_.update(train_cfg=train_cfg)
            semseg_head_.update(test_cfg=test_cfg)
            self.sem_seg_head = MODELS.build(semseg_head_)

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        visual_features = self.extract_feat(batch_inputs)

        if self.sem_seg_head:
            outputs = self.sem_seg_head.predict(
                visual_features,
                batch_data_samples,
                rescale=rescale)
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]

            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(batch_inputs.shape[-2], batch_inputs.shape[-1]),
                mode="bicubic",
                align_corners=False,
                antialias=True
            )

            for mask_cls_result, mask_pred_result, img_metas, data_samples in zip(mask_cls_results, mask_pred_results,
                                                                                  batch_img_metas, batch_data_samples):
                height = img_metas["ori_shape"][0]
                width = img_metas["ori_shape"][1]
                image_size = img_metas["img_shape"][:2]

                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width
                )  # 101，h,w,
                mask_cls_result = mask_cls_result.to(mask_pred_result)  # 101,10，

                sem_seg = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result, False)
                if sem_seg.shape[0] == 1:
                    sem_seg = sem_seg.squeeze(0) > 0.5
                else:
                    sem_seg = sem_seg.max(0)[1]
                pred_sem_seg = PixelData(metainfo={'bg_value': 0}, data=sem_seg)
                data_samples.pred_sem_seg = pred_sem_seg

        return batch_data_samples

    def semantic_inference(self, mask_cls, mask_pred, keep_sem_bgd=False):
        if keep_sem_bgd:
            mask_cls = F.softmax(mask_cls, dim=-1)
        else:
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg
