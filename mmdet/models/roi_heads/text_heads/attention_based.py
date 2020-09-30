"""
 Copyright (c) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import HEADS


class Encoder(nn.Module):
    def __init__(self, dim_input, dim_internal, num_layers):
        super().__init__()

        self.dim_input = dim_input

        module_list = []
        for i in range(num_layers):
            module_list.extend([
                nn.Conv2d(dim_input, dim_internal,
                          kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(dim_internal),
                nn.ReLU(inplace=True)
            ])
            dim_input = dim_internal
        self.layers = nn.Sequential(*module_list)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, feature):
        feature = self.layers(feature)
        return feature


class DecoderAttention2d(nn.Module):
    str_to_class = {
        'GRU': nn.GRU,
        'LSTM': nn.LSTM
    }

    def __init__(self, hidden_size, vocab_size, decoder_input_feature_size, rnn_type):
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        assert len(decoder_input_feature_size) == 2
        self.flatten_feature_size = decoder_input_feature_size[0] * \
            decoder_input_feature_size[1]

        self.embedding = nn.Embedding(vocab_size, self.hidden_size)

        self.decoder = self.str_to_class[rnn_type](
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1, bidirectional=False)

        self.encoder_outputs_w = nn.Linear(self.hidden_size, self.hidden_size)
        self.hidden_state_w = nn.Linear(self.hidden_size, self.hidden_size)
        self.v = nn.Parameter(torch.Tensor(
            self.hidden_size, 1))  # context vector

        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.out = nn.Linear(self.hidden_size, self.vocab_size)

        nn.init.normal_(self.v, 0, 0.1)

    def forward(self, input, hidden, encoder_outputs, cell=None):
        '''

        :param input: Shape is [1, BATCH_SIZE]
        :param hidden: Shape is [1, BATCH_SIZE, HIDDEN_DIM]
        :param cell: Shape is [1, BATCH_SIZE, HIDDEN_DIM], it is used in case of LSTM
        :param encoder_outputs: [BATCH_SIZE, T, HIDDEN_DIM]
        :return:
        '''
        BATCH_SIZE = hidden.shape[1]
        assert tuple(hidden.shape) == (1, BATCH_SIZE, self.hidden_size), f'{hidden.shape}'
        assert tuple(input.shape) == (BATCH_SIZE,), f'{input.shape} {input}'
        assert tuple(encoder_outputs.shape) == (
            BATCH_SIZE, self.flatten_feature_size, self.hidden_size), f'{encoder_outputs.shape}'

        input = input.long()

        input = self.embedding(input)

        encoder_outputs_w = self.encoder_outputs_w(encoder_outputs)
        hidden_state_w = self.hidden_state_w(hidden[0]).unsqueeze(1)
        assert tuple(hidden_state_w.shape) == (BATCH_SIZE, 1, self.hidden_size)
        hidden_state_w = hidden_state_w.expand(
            (BATCH_SIZE, encoder_outputs_w.shape[1], self.hidden_size))
        assert tuple(hidden_state_w.shape) == (
            BATCH_SIZE, encoder_outputs_w.shape[1], self.hidden_size)

        s = torch.tanh(encoder_outputs_w + hidden_state_w)
        assert tuple(s.shape) == (
            BATCH_SIZE, self.flatten_feature_size, self.hidden_size)
        s = s.reshape(-1, self.hidden_size)
        s = torch.matmul(s, self.v)
        s = s.reshape(-1, self.flatten_feature_size)

        attn_weights = F.softmax(s, dim=1)
        # print('attn_weights.shape', attn_weights.shape)

        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        # print('attn_applied.shape', attn_applied.shape)

        attn_applied = attn_applied.permute(1, 0, 2)
        attn_applied = attn_applied.squeeze(0)

        output = torch.cat((input, attn_applied), 1)

        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        if isinstance(self.decoder, nn.GRU):
            output, hidden = self.decoder(output, hidden)
        elif isinstance(self.decoder, nn.LSTM):
            output, (hidden, cell) = self.decoder(output, (hidden, cell))

        # to avoid removing/renaming output by mo.py
        hidden = torch.reshape(hidden, hidden.shape)

        output = self.out(output[0])
        if self.training:
            output = F.log_softmax(output, dim=1)

        if isinstance(self.decoder, nn.LSTM):
            return output, hidden, cell, attn_weights
        if isinstance(self.decoder, nn.GRU):
            return output, hidden, attn_weights


@HEADS.register_module()
class TextRecognitionHeadAttention(nn.Module):

    def __init__(self,
                 input_feature_size,
                 encoder_dim_input,
                 encoder_dim_internal,
                 encoder_num_layers,
                 decoder_input_feature_size,
                 decoder_max_seq_len,
                 decoder_vocab_size,
                 decoder_dim_hidden,
                 decoder_sos_index,
                 decoder_rnn_type,
                 visualize):
        super().__init__()

        self.input_feature_size = input_feature_size
        self.encoder_dim_input = encoder_dim_input

        self.encoder = Encoder(
            encoder_dim_input, encoder_dim_internal, encoder_num_layers)
        self.dropout = nn.Dropout(0.5)
        self.decoder = DecoderAttention2d(hidden_size=decoder_dim_hidden,
                                          vocab_size=decoder_vocab_size,
                                          decoder_input_feature_size=decoder_input_feature_size,
                                          rnn_type=decoder_rnn_type)

        self.decoder_input_feature_size = decoder_input_feature_size
        self.decoder_max_seq_len = decoder_max_seq_len
        self.decoder_sos_int = decoder_sos_index
        self.decoder_dim_hidden = decoder_dim_hidden

        self.visualize = visualize

        self.criterion = nn.NLLLoss(reduction='none')

    def __forward_train(self, features, targets, masks):
        if not all([len(target) <= self.decoder_max_seq_len for target in targets if len(target)]):
            return torch.tensor(0.0, device=features.device)

        valid_targets_indexes = torch.tensor([ind for ind, target in enumerate(targets) if len(target)], device=features.device)

        if len(valid_targets_indexes) == 0:
            return torch.tensor(0.0, device=features.device)
        targets = [np.array(target) for target in targets if len(target)]
        targets = [np.pad(target, (0, self.decoder_max_seq_len - len(target))) for target in targets]
        targets = np.array(targets)

        batch_size = targets.shape[0]

        features = features[valid_targets_indexes]
        features = features.view(features.shape[0], features.shape[1], -1)  # B C H*W
        features = features.permute(0, 2, 1)  # BxH*WxC or BxTxC
        features = self.dropout(features)

        decoder_hidden = torch.zeros([1, batch_size, self.decoder_dim_hidden], device=features.device)
        decoder_cell = torch.zeros([1, batch_size, self.decoder_dim_hidden], device=features.device)
        loss = 0
        positive_counter = 0
        decoder_input = torch.ones([batch_size], device=features.device, dtype=torch.long) * self.decoder_sos_int
        targets = torch.tensor(targets, device=features.device, dtype=torch.long)

        

        predictions = []

        for di in range(self.decoder_max_seq_len):
            if isinstance(self.decoder.decoder, nn.GRU):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, features)
            elif isinstance(self.decoder.decoder, nn.LSTM):
                decoder_output, decoder_hidden, decoder_cell, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, features, decoder_cell)
            predictions.append(decoder_output.topk(1)[1].cpu().numpy().reshape(-1))
            # print(targets[:, di].reshape(-1))
            # print('---')
            mask = (targets[:, di] != 0).float()
            loss += self.criterion(decoder_output, targets[:, di]) * mask
            mask_sum = torch.sum(mask)
            if mask_sum == 0:
                break
            positive_counter += mask_sum
            decoder_input = targets[:, di]

        # predictions = np.transpose(predictions)
        # print(targets.cpu().numpy()[:, :predictions.shape[1]])
        # print(predictions)
        # print('----------------')
        assert positive_counter > 0
        loss = torch.sum(loss) / positive_counter
        return loss.to(features.device)

    def __forward_test(self, features, masks):
        batch_size = features.shape[0]
        features = features.view(features.shape[0], features.shape[1], -1)
        features = features.permute(0, 2, 1)
        decoder_hidden = torch.zeros([1, batch_size, self.decoder_dim_hidden],
                                     device=features.device)
        decoder_cell = torch.zeros([1, batch_size, self.decoder_dim_hidden],
                                   device=features.device)
        decoder_input = torch.ones([batch_size], device=features.device,
                                   dtype=torch.long) * self.decoder_sos_int
        decoder_outputs = []
        if self.visualize:
            full_attention_mask = np.zeros([112, 112], dtype=np.uint8)
        for di in range(self.decoder_max_seq_len):
            if isinstance(self.decoder.decoder, nn.GRU):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, features)
            elif isinstance(self.decoder.decoder, nn.LSTM):
                decoder_output, decoder_hidden, decoder_cell, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, features, decoder_cell)
            if self.visualize:
                attention = decoder_attention.cpu().detach().numpy()
                attention = (np.reshape(attention, (-1, 28))
                             * 500).astype(np.uint8)
                attention = cv2.resize(attention, (112, 112))
                full_attention_mask += attention
                cv2.imshow('attention', attention)
                cv2.waitKey(30)
            topv, topi = decoder_output.topk(1)
            decoder_outputs.append(decoder_output)
            decoder_input = topi.detach().view(batch_size)
        decoder_outputs = torch.stack(decoder_outputs)
        if self.visualize:
            cv2.imshow('full', full_attention_mask)
        return decoder_outputs

    def forward(self, features, target=None, masks=None):
        features = self.encoder(features)
        if masks is not None:
            masks = masks.expand(-1, features.shape[1], -1, -1)
            features = features * masks

        if self.training:
            return self.__forward_train(features, target, masks)
        else:
            return self.__forward_test(features, masks)

    def dummy_forward(self):
        return torch.zeros((1, self.decoder_max_seq_len, self.decoder.vocab_size),
                           dtype=torch.float32)
