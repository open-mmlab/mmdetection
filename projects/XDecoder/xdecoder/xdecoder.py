from torch import Tensor

from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class XDecoder(SingleStageDetector):

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super(SingleStageDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)

        head_ = head.deepcopy()
        head_.update(test_cfg=test_cfg)
        self.sem_seg_head = MODELS.build(head_)  # TODO: sem_seg_head -> head

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        visual_features = self.extract_feat(batch_inputs)
        outputs = self.sem_seg_head.predict(visual_features,
                                            batch_data_samples)
        return outputs
