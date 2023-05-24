from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.single_stage import SingleStageDetector


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

        self.semseg_head = None
        if semseg_head is not None:
            semseg_head_ = semseg_head.deepcopy()
            semseg_head_.update(train_cfg=train_cfg)
            semseg_head_.update(test_cfg=test_cfg)
            self.semseg_head = MODELS.build(semseg_head_)

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:

        visual_features = self.extract_feat(batch_inputs)

        if self.semseg_head:
            results_list = self.semseg_head.predict(
                visual_features,
                batch_data_samples,
                rescale=rescale)
        return results_list
