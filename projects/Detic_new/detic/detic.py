# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

from mmdet.datasets import LVISV1Dataset
from mmdet.models.detectors.cascade_rcnn import CascadeRCNN
from mmdet.registry import MODELS
from mmdet.structures import SampleList


class CLIPTextEncoder(nn.Module):

    def __init__(self, model_name='ViT-B/32'):
        super().__init__()
        import clip
        from clip.simple_tokenizer import SimpleTokenizer
        self.tokenizer = SimpleTokenizer()
        pretrained_model, _ = clip.load(model_name, device='cpu')
        self.clip = pretrained_model

    @property
    def device(self):
        return self.clip.device

    @property
    def dtype(self):
        return self.clip.dtype

    def tokenize(self,
                 texts: Union[str, List[str]],
                 context_length: int = 77) -> torch.LongTensor:
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.tokenizer.encoder['<|startoftext|>']
        eot_token = self.tokenizer.encoder['<|endoftext|>']
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token]
                      for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                st = torch.randint(len(tokens) - context_length + 1,
                                   (1, ))[0].item()
                tokens = tokens[st:st + context_length]
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    def forward(self, text):
        text = self.tokenize(text)
        text_features = self.clip.encode_text(text)
        return text_features


def get_class_weight(original_caption, prompt_prefix='a '):
    if isinstance(original_caption, str):
        if original_caption == 'coco':
            from mmdet.datasets import CocoDataset
            class_names = CocoDataset.METAINFO['classes']
        elif original_caption == 'cityscapes':
            from mmdet.datasets import CityscapesDataset
            class_names = CityscapesDataset.METAINFO['classes']
        elif original_caption == 'voc':
            from mmdet.datasets import VOCDataset
            class_names = VOCDataset.METAINFO['classes']
        elif original_caption == 'openimages':
            from mmdet.datasets import OpenImagesDataset
            class_names = OpenImagesDataset.METAINFO['classes']
        elif original_caption == 'lvis':
            from mmdet.datasets import LVISV1Dataset
            class_names = LVISV1Dataset.METAINFO['classes']
        else:
            if not original_caption.endswith('.'):
                original_caption = original_caption + ' . '
            original_caption = original_caption.split(' . ')
            class_names = list(filter(lambda x: len(x) > 0, original_caption))

    # for test.py
    else:
        class_names = list(original_caption)

    text_encoder = CLIPTextEncoder()
    text_encoder.eval()
    texts = [prompt_prefix + x for x in class_names]
    print_log(f'Computing text embeddings for {len(class_names)} classes.')
    embeddings = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return class_names, embeddings


def reset_cls_layer_weight(roi_head, weight):
    if type(weight) == str:
        print_log(f'Resetting cls_layer_weight from file: {weight}')
        zs_weight = torch.tensor(
            np.load(weight),
            dtype=torch.float32).permute(1, 0).contiguous()  # D x C
    else:
        zs_weight = weight
    zs_weight = torch.cat(
        [zs_weight, zs_weight.new_zeros(
            (zs_weight.shape[0], 1))], dim=1)  # D x (C + 1)
    zs_weight = F.normalize(zs_weight, p=2, dim=0)
    zs_weight = zs_weight.to('cuda')
    num_classes = zs_weight.shape[-1]

    for bbox_head in roi_head.bbox_head:
        bbox_head.num_classes = num_classes
        del bbox_head.fc_cls.zs_weight
        bbox_head.fc_cls.zs_weight = zs_weight


@MODELS.register_module()
class Detic(CascadeRCNN):

    def __init__(self,
                 with_image_labels: bool = False,
                 sync_caption_batch: bool = False,
                 fp16: bool = False,
                 roi_head_name: str = '',
                 cap_batch_ratio: int = 4,
                 with_caption: bool = False,
                 dynamic_classifier: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self._entities = LVISV1Dataset.METAINFO['classes']
        self._text_prompts = None
        # Turn on co-training with classification data
        self.with_image_labels = with_image_labels
        # Caption losses
        self.with_caption = with_caption
        # synchronize across GPUs to enlarge # "classes"
        self.sync_caption_batch = sync_caption_batch
        # Ratio between detection data and caption data
        self.cap_batch_ratio = cap_batch_ratio
        self.fp16 = fp16
        self.roi_head_name = roi_head_name
        # dynamic class sampling when training with 21K classes,
        # Federated loss is enabled when DYNAMIC_CLASSIFIER is on
        self.dynamic_classifier = dynamic_classifier
        self.return_proposal = False
        if self.dynamic_classifier:
            self.freq_weight = kwargs.pop('freq_weight')
            self.num_classes = kwargs.pop('num_classes')
            self.num_sample_cats = kwargs.pop('num_sample_cats')

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """

        x = self.extract_feat(batch_inputs)
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)

            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
            # if not hasattr(batch_data_samples[0].gt_instances, 'bboxes'):
            #     losses.update({k: v * 0 for k, v in rpn_losses.items()})
            # else:
            #     losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)

        losses.update(roi_losses)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        # For single image inference
        if 'custom_entities' in batch_data_samples[0]:
            text_prompts = batch_data_samples[0].text
            if text_prompts != self._text_prompts:
                self._text_prompts = text_prompts
                class_names, zs_weight = get_class_weight(text_prompts)
                self._entities = class_names
                reset_cls_layer_weight(self.roi_head, zs_weight)

        assert self.with_bbox, 'Bbox head must be implemented.'

        x = self.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        for data_sample, pred_instances in zip(batch_data_samples,
                                               results_list):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    label_names.append(self._entities[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances

        return batch_data_samples
