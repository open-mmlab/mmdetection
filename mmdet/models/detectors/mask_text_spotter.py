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

    def export(self, img, img_metas, f='', **kwargs):

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

        with self.forward_export_context(img_metas):
            torch.onnx.export(self, img, f, **kwargs)

        # Export of text recognition encoder
        export_to_onnx_text_recognition_encoder(
            self.roi_head.text_head.encoder,
            self.roi_head.text_head.input_feature_size,
            f.replace('.onnx', '_text_recognition_head_encoder.onnx')
        )

        # Export of text recognition decoder
        export_to_onnx_text_recognition_decoder(
            self.roi_head.text_head.decoder,
            self.roi_head.text_head.input_feature_size,
            f.replace('.onnx', '_text_recognition_head_decoder.onnx')
        )
