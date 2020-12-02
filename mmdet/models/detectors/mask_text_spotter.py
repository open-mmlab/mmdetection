import os

import numpy as np
import onnx
import torch
import torch.nn as nn

from ..builder import DETECTORS
from .mask_rcnn import MaskRCNN


@DETECTORS.register_module()
class MaskTextSpotter(MaskRCNN):

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(MaskTextSpotter, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_texts=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            gt_texts (None | Tensor) : true texts for each box
                used if the architecture supports a text task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                gt_masks=gt_masks)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 gt_texts ,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses


    def export(self, img, img_metas, export_name='', **kwargs):

        def export_to_onnx_text_recognition_decoder(net, input_size, path_to_onnx):
            net.eval()

            dim = net.hidden_size
            prev_input = np.random.randn(1).astype(np.float32)
            pev_hidden = np.random.randn(1, 1, dim).astype(np.float32)
            prev_cell = np.random.randn(1, 1, dim).astype(np.float32)
            encoder_outputs = np.random.randn(1, input_size[0] * input_size[1], dim).astype(np.float32)
            prev_input = torch.tensor(prev_input)
            pev_hidden = torch.tensor(pev_hidden)
            prev_cell = torch.tensor(prev_cell)
            encoder_outputs = torch.tensor(encoder_outputs)
            if torch.cuda.is_available():
                net = net.cuda()
                prev_input = prev_input.cuda()
                pev_hidden = pev_hidden.cuda()
                prev_cell = prev_cell.cuda()
                encoder_outputs = encoder_outputs.cuda()
            # prev_input.requires_grad = False
            # pev_hidden.requires_grad = True
            # prev_cell.requires_grad = True
            # encoder_outputs.requires_grad = True
            if isinstance(net.decoder, nn.GRU):
                torch.onnx.export(net, (prev_input, pev_hidden, encoder_outputs),
                                  path_to_onnx, verbose=True,
                                  input_names=['prev_symbol', 'prev_hidden', 'encoder_outputs'],
                                  output_names=['output', 'hidden', 'attention']
                                  )
            elif isinstance(net.decoder, nn.LSTM):
                torch.onnx.export(net, (prev_input, pev_hidden, encoder_outputs, prev_cell),
                                  path_to_onnx, verbose=True,
                                  input_names=['prev_symbol', 'prev_hidden', 'encoder_outputs', 'prev_cell'],
                                  output_names=['output', 'hidden', 'cell', 'attention']
                                  )

            printable_graph = onnx.helper.printable_graph(onnx.load(path_to_onnx).graph)
            print(printable_graph)

            return printable_graph


        def export_to_onnx_text_recognition_encoder(net, input_size, path_to_onnx):
            net.eval()
            dim = net.dim_input
            input = np.random.randn(1, dim, *input_size).astype(np.float32)
            input = torch.tensor(input)
            if torch.cuda.is_available():
                net = net.cuda()
                input = input.cuda()
            input.requires_grad = True
            torch.onnx.export(net, input, path_to_onnx, verbose=True,
                              input_names=['input'], output_names=['output'])

            printable_graph = onnx.helper.printable_graph(onnx.load(path_to_onnx).graph)
            print(printable_graph)

            return printable_graph

        self.img_metas = img_metas
        self.forward_backup = self.forward
        self.forward = self.forward_export
        torch.onnx.export(self, img, export_name, **kwargs)
        self.forward = self.forward_backup

        # Export of text recognition encoder
        export_to_onnx_text_recognition_encoder(
            self.roi_head.text_head.encoder,
            self.roi_head.text_head.input_feature_size,
            export_name.replace('.onnx', '_text_recognition_head_encoder.onnx')
        )

        # Export of text recognition decoder
        export_to_onnx_text_recognition_decoder(
            self.roi_head.text_head.decoder,
            self.roi_head.text_head.input_feature_size,
            export_name.replace('.onnx', '_text_recognition_head_decoder.onnx')
        )
