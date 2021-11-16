from ..builder import DETECTORS
import torch
from torch import nn
from ..builder import build_backbone, build_head
# from .single_stage import SingleStageDetector
from mmcv.runner import BaseModule, auto_fp16
from collections import OrderedDict 
import torch.distributed as dist
from mmdet.models.backbones import build_resnet_backbone


@DETECTORS.register_module()
class MaskFormer(nn.Module):
    # one-stage
    def __init__(self,
                 backbone=None,
                 sem_seg_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(MaskFormer, self).__init__()

        self.backbone = build_resnet_backbone()

        # if sem_seg_head is not None:
        self.sem_seg_head = build_head(sem_seg_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg


    @property
    def with_sem_seg_head(self):
        return (hasattr(self, 'sem_seg_head') and self.sem_seg_head is not None)


    def extract_feat(self, imgs):
        return self.backbone(imgs)


    def extract_feats(self, imgs):
        """Extract features from multiple images.

        Args:
            imgs (list[torch.Tensor]): A list of images. The images are
                augmented from the same image but in different ways.

        Returns:
            list[torch.Tensor]: Features of different images
        """
        assert isinstance(imgs, list)
        return [self.extract_feat(img) for img in imgs]


    def forward_train(self, 
                      img, 
                      img_metas, 
                      gt_bboxes, 
                      gt_labels, 
                      gt_masks, 
                      gt_semantic_seg,
                      gt_bboxes_ignore=None,
                      **kargs):
        """
        Args:
            img (Tensor): shape = (B, C, H, W).
            img_metas (list[Dict]): image metas.
            gt_bboxes (list[Tensor]): shape of each element = (num_objects, 4).
            gt_labels (list[Tensor]): shape of each element = (num_objects, ).
            gt_masks (list[BitmapMasks]): 表示不同的实例的mask.
            gt_semantic_seg (list[tensor]): thing 在前 0-79， stuff在后 80-132, 255为背景. 
            gt_bboxes_ignore (list[Tensor]): # TODO 不知道是否需要考虑掉这些实例. Defaults to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)
        losses = self.sem_seg_head.forward_train(x, img_metas, 
                                                        gt_bboxes, gt_labels, gt_bboxes_ignore, 
                                                        gt_masks, gt_semantic_seg)
        
        return losses


    def simple_test(self, img, img_metas, **kwargs):

        # if img_metas[0]["ori_filename"] == "000000000139.jpg":
        #     print(img_metas[0]["ori_shape"])
        #     print(img_metas[0]["img_shape"])
        #     print(img_metas[0]["pad_shape"])
        #     print(img[0].shape)
        #     # (426, 640, 3)
            # (800, 1202, 3)
            # (800, 1216, 3)
            # torch.Size([3, 800, 1216])

        feat = self.extract_feat(img)
        mask_results = self.sem_seg_head.simple_test(
            feat, img_metas, **kwargs)
        # TODO convert results_list to mask_results
        # mask_results = None
        return mask_results


    def aug_test(self, imgs, img_metas, **kwargs):
        raise NotImplementedError


    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)


    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])

        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)


    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars


    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
        
        return outputs


    def val_step(self, data, optimizer=None):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs


    def show_result(self):
        # TODO plot the panoptic segmentation result
        pass


    def onnx_export(self, img, img_metas):
        raise NotImplementedError


    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool = True):
        new_state_dict = {}
        for k in state_dict.keys():
            if "criterion" in k:
                new_state_dict["sem_seg_head.criterion.empty_weight"] = state_dict[k]
            else:
                new_state_dict[k] = state_dict[k]

        return super().load_state_dict(new_state_dict, strict=strict)