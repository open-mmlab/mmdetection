# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Union

from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.structures.bbox import BaseBoxes
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector

try:
    # TODO: Whether need to check d2 version
    import detectron2
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
except ImportError:
    detectron2 = None


def _to_cfgnode_list(cfg: ConfigType,
                     config_list: list = [],
                     father_name: str = 'MODEL') -> tuple:
    """Convert the key and value of mmengine.ConfigDict into a list.

    Args:
        cfg (ConfigDict): The detectron2 model config.
        config_list (list): A list contains the key and value of ConfigDict.
            Defaults to [].
        father_name (str): The father name add before the key.
            Defaults to "MODEL".

    Returns:
        tuple:

            - config_list: A list contains the key and value of ConfigDict.
            - father_name (str): The father name add before the key.
              Defaults to "MODEL".
    """
    for key, value in cfg.items():
        name = f'{father_name}.{key.upper()}'
        if isinstance(value, ConfigDict) or isinstance(value, dict):
            config_list, fater_name = \
                _to_cfgnode_list(value, config_list, name)
        else:
            config_list.append(name)
            config_list.append(value)

    return config_list, father_name


def add_d2_pred_to_datasample(data_samples: SampleList,
                              d2_results_list: list) -> SampleList:
    """"""
    assert len(data_samples) == len(d2_results_list)
    for data_sample, d2_results in zip(data_samples, d2_results_list):
        d2_instance = d2_results['instances']

        results = InstanceData()
        results.bboxes = d2_instance.pred_boxes.tensor
        results.scores = d2_instance.scores
        results.labels = d2_instance.pred_classes

        if d2_instance.has('pred_masks'):
            results.masks = d2_instance.pred_masks
        data_sample.pred_instances = results

    return data_samples


@MODELS.register_module()
class Detectron2Wrapper(BaseDetector):

    def __init__(self,
                 d2_detector: ConfigType,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        if detectron2 is None:
            raise ImportError('Please install detectron2 first')
        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        if data_preprocessor is not None:
            data_preprocessor = None
            warnings.warn('The `data_preprocessor` should be None.')
        if init_cfg is not None:
            init_cfg = None
            warnings.warn('The `init_cfg` should be None.')
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self._channel_conversion = rgb_to_bgr or bgr_to_rgb
        cfgnode_list, _ = _to_cfgnode_list(d2_detector)
        self.cfg = get_cfg()
        self.cfg.merge_from_list(cfgnode_list)
        self.d2_model = build_model(self.cfg)

    def init_weights(self) -> None:
        """Initialization Backbone.

        NOTE: The initialization of other layers are in detectron2,
        if users want to change the initialization way, please
        change the code in detectron2.
        """
        from detectron2.checkpoint import DetectionCheckpointer
        checkpointer = DetectionCheckpointer(model=self.d2_model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS, checkpointables=[])

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, tuple]:
        """Calculate losses from a batch of inputs and data samples."""
        print()
        d2_batched_inputs = self._convert_to_batched_d2_inputs(
            batch_inputs=batch_inputs, batch_data_samples=batch_data_samples)
        from detectron2.utils.events import EventStorage
        with EventStorage() as storage:  # noqa
            losses = self.d2_model(d2_batched_inputs)
        # storage contains some training information, such as cls_accuracy.
        # you can use storage.latest() to get the detail information
        return losses

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        d2_batched_inputs = self._convert_to_batched_d2_inputs(
            batch_inputs=batch_inputs, batch_data_samples=batch_data_samples)
        # results in detectron2 has already rescale
        d2_results_list = self.d2_model(d2_batched_inputs)
        batch_data_samples = add_d2_pred_to_datasample(
            data_samples=batch_data_samples, d2_results_list=d2_results_list)

        return batch_data_samples

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        raise NotImplementedError(
            f'`_forward` is not implemented in {self.__class__.__name__}')

    def extract_feat(self, batch_inputs: Tensor):
        """Extract features from images.

        `extract_feat` will not be used in obj:``Detectron2Wrapper``.
        """
        pass

    def _convert_to_batched_d2_inputs(self, batch_inputs: Tensor,
                                      batch_data_samples: SampleList) -> list:
        from detectron2.data.detection_utils import filter_empty_instances
        from detectron2.structures import Boxes, Instances

        batched_d2_inputs = []
        for image, data_samples in zip(batch_inputs, batch_data_samples):
            d2_inputs = dict()
            # deal with metainfo
            meta_info = data_samples.metainfo
            d2_inputs['file_name'] = meta_info['img_path']
            d2_inputs['height'], d2_inputs['width'] = meta_info['ori_shape']
            d2_inputs['image_id'] = meta_info['img_id']
            # deal with image
            if self._channel_conversion:
                image = image[[2, 1, 0], ...]
            d2_inputs['image'] = image
            # deal with gt_instances
            gt_instances = data_samples.gt_instances
            d2_instances = Instances(meta_info['ori_shape'])

            gt_boxes = gt_instances.bboxes
            if isinstance(gt_boxes, BaseBoxes):
                gt_boxes = gt_boxes.tensor
            d2_instances.gt_boxes = Boxes(gt_boxes)

            d2_instances.gt_classes = gt_instances.labels
            if gt_instances.get('masks', None) is not None:
                d2_instances.gt_masks = gt_instances.masks
            d2_inputs['instances'] = filter_empty_instances(d2_instances)
            batched_d2_inputs.append(d2_inputs)

        return batched_d2_inputs
