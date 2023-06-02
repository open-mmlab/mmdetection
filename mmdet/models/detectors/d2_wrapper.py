# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import BaseBoxes
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from mmdet.utils import ConfigType
from .base import BaseDetector

try:
    import detectron2
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from detectron2.structures.masks import BitMasks as D2_BitMasks
    from detectron2.structures.masks import PolygonMasks as D2_PolygonMasks
    from detectron2.utils.events import EventStorage
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


def convert_d2_pred_to_datasample(data_samples: SampleList,
                                  d2_results_list: list) -> SampleList:
    """Convert the Detectron2's result to DetDataSample.

    Args:
        data_samples (list[:obj:`DetDataSample`]): The batch
            data samples. It usually includes information such
            as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
        d2_results_list (list): The list of the results of Detectron2's model.

    Returns:
        list[:obj:`DetDataSample`]: Detection results of the
        input images. Each DetDataSample usually contain
        'pred_instances'. And the ``pred_instances`` usually
        contains following keys.

        - scores (Tensor): Classification scores, has a shape
          (num_instance, )
        - labels (Tensor): Labels of bboxes, has a shape
          (num_instances, ).
        - bboxes (Tensor): Has a shape (num_instances, 4),
          the last dimension 4 arrange as (x1, y1, x2, y2).
    """
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
    """Wrapper of a Detectron2 model. Input/output formats of this class follow
    MMDetection's convention, so a Detectron2 model can be trained and
    evaluated in MMDetection.

    Args:
        detector (:obj:`ConfigDict` or dict): The module config of
            Detectron2.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to BGR.
            Defaults to False.
    """

    def __init__(self,
                 detector: ConfigType,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False) -> None:
        if detectron2 is None:
            raise ImportError('Please install Detectron2 first')
        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        super().__init__()
        self._channel_conversion = rgb_to_bgr or bgr_to_rgb
        cfgnode_list, _ = _to_cfgnode_list(detector)
        self.cfg = get_cfg()
        self.cfg.merge_from_list(cfgnode_list)
        self.d2_model = build_model(self.cfg)
        self.storage = EventStorage()

    def init_weights(self) -> None:
        """Initialization Backbone.

        NOTE: The initialization of other layers are in Detectron2,
        if users want to change the initialization way, please
        change the code in Detectron2.
        """
        from detectron2.checkpoint import DetectionCheckpointer
        checkpointer = DetectionCheckpointer(model=self.d2_model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS, checkpointables=[])

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, tuple]:
        """Calculate losses from a batch of inputs and data samples.

        The inputs will first convert to the Detectron2 type and feed into
        D2 models.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        d2_batched_inputs = self._convert_to_d2_inputs(
            batch_inputs=batch_inputs,
            batch_data_samples=batch_data_samples,
            training=True)

        with self.storage as storage:  # noqa
            losses = self.d2_model(d2_batched_inputs)
        # storage contains some training information, such as cls_accuracy.
        # you can use storage.latest() to get the detail information
        return losses

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        The inputs will first convert to the Detectron2 type and feed into
        D2 models. And the results will convert back to the MMDet type.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.


        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        d2_batched_inputs = self._convert_to_d2_inputs(
            batch_inputs=batch_inputs,
            batch_data_samples=batch_data_samples,
            training=False)
        # results in detectron2 has already rescale
        d2_results_list = self.d2_model(d2_batched_inputs)
        batch_data_samples = convert_d2_pred_to_datasample(
            data_samples=batch_data_samples, d2_results_list=d2_results_list)

        return batch_data_samples

    def _forward(self, *args, **kwargs):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        raise NotImplementedError(
            f'`_forward` is not implemented in {self.__class__.__name__}')

    def extract_feat(self, *args, **kwargs):
        """Extract features from images.

        `extract_feat` will not be used in obj:``Detectron2Wrapper``.
        """
        pass

    def _convert_to_d2_inputs(self,
                              batch_inputs: Tensor,
                              batch_data_samples: SampleList,
                              training=True) -> list:
        """Convert inputs type to support Detectron2's model.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            training (bool): Whether to enable training time processing.

        Returns:
            list[dict]: A list of dict, which will be fed into Detectron2's
            model. And the dict usually contains following keys.

            - image (Tensor): Image in (C, H, W) format.
            - instances (Instances): GT Instance.
            - height (int): the output height resolution of the model
            - width (int): the output width resolution of the model
        """
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
            d2_instances = Instances(meta_info['img_shape'])

            gt_boxes = gt_instances.bboxes
            # TODO: use mmdet.structures.box.get_box_tensor after PR 8658
            #  has merged
            if isinstance(gt_boxes, BaseBoxes):
                gt_boxes = gt_boxes.tensor
            d2_instances.gt_boxes = Boxes(gt_boxes)

            d2_instances.gt_classes = gt_instances.labels
            if gt_instances.get('masks', None) is not None:
                gt_masks = gt_instances.masks
                if isinstance(gt_masks, PolygonMasks):
                    d2_instances.gt_masks = D2_PolygonMasks(gt_masks.masks)
                elif isinstance(gt_masks, BitmapMasks):
                    d2_instances.gt_masks = D2_BitMasks(gt_masks.masks)
                else:
                    raise TypeError('The type of `gt_mask` can be '
                                    '`PolygonMasks` or `BitMasks`, but get '
                                    f'{type(gt_masks)}.')
            # convert to cpu and convert back to cuda to avoid
            # some potential error
            if training:
                device = gt_boxes.device
                d2_instances = filter_empty_instances(
                    d2_instances.to('cpu')).to(device)
                d2_inputs['instances'] = d2_instances
            batched_d2_inputs.append(d2_inputs)

        return batched_d2_inputs
