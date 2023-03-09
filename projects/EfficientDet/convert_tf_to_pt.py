import argparse

import numpy as np
import torch
from tensorflow.python.training import py_checkpoint_reader

torch.set_printoptions(precision=20)


def tf2pth(v):
    if v.ndim == 4:
        return np.ascontiguousarray(v.transpose(3, 2, 0, 1))
    elif v.ndim == 2:
        return np.ascontiguousarray(v.transpose())
    return v


def convert_key(model_name, bifpn_repeats, weights):

    p6_w1 = [
        torch.tensor([-1e4, -1e4], dtype=torch.float64)
        for _ in range(bifpn_repeats)
    ]
    p5_w1 = [
        torch.tensor([-1e4, -1e4], dtype=torch.float64)
        for _ in range(bifpn_repeats)
    ]
    p4_w1 = [
        torch.tensor([-1e4, -1e4], dtype=torch.float64)
        for _ in range(bifpn_repeats)
    ]
    p3_w1 = [
        torch.tensor([-1e4, -1e4], dtype=torch.float64)
        for _ in range(bifpn_repeats)
    ]
    p4_w2 = [
        torch.tensor([-1e4, -1e4, -1e4], dtype=torch.float64)
        for _ in range(bifpn_repeats)
    ]
    p5_w2 = [
        torch.tensor([-1e4, -1e4, -1e4], dtype=torch.float64)
        for _ in range(bifpn_repeats)
    ]
    p6_w2 = [
        torch.tensor([-1e4, -1e4, -1e4], dtype=torch.float64)
        for _ in range(bifpn_repeats)
    ]
    p7_w2 = [
        torch.tensor([-1e4, -1e4], dtype=torch.float64)
        for _ in range(bifpn_repeats)
    ]
    idx2key = {
        0: '1.0',
        1: '2.0',
        2: '2.1',
        3: '3.0',
        4: '3.1',
        5: '4.0',
        6: '4.1',
        7: '4.2',
        8: '4.3',
        9: '4.4',
        10: '4.5',
        11: '5.0',
        12: '5.1',
        13: '5.2',
        14: '5.3',
        15: '5.4'
    }
    m = dict()
    for k, v in weights.items():

        if 'Exponential' in k or 'global_step' in k:
            continue

        seg = k.split('/')
        if len(seg) == 1:
            continue
        if seg[2] == 'depthwise_conv2d':
            v = v.transpose(1, 0)

        if seg[0] == model_name:
            if seg[1] == 'stem':
                prefix = 'backbone.layers.0'
                mapping = {
                    'conv2d/kernel': 'conv.weight',
                    'tpu_batch_normalization/beta': 'bn.bias',
                    'tpu_batch_normalization/gamma': 'bn.weight',
                    'tpu_batch_normalization/moving_mean': 'bn.running_mean',
                    'tpu_batch_normalization/moving_variance':
                    'bn.running_var',
                }
                suffix = mapping['/'.join(seg[2:])]
                m[prefix + '.' + suffix] = v

            elif seg[1].startswith('blocks_'):
                idx = int(seg[1][7:])
                prefix = '.'.join(['backbone', 'layers', idx2key[idx]])
                base_mapping = {
                    'depthwise_conv2d/depthwise_kernel':
                    'depthwise_conv.conv.weight',
                    'se/conv2d/kernel': 'se.conv1.conv.weight',
                    'se/conv2d/bias': 'se.conv1.conv.bias',
                    'se/conv2d_1/kernel': 'se.conv2.conv.weight',
                    'se/conv2d_1/bias': 'se.conv2.conv.bias'
                }
                if idx == 0:
                    mapping = {
                        'conv2d/kernel':
                        'linear_conv.conv.weight',
                        'tpu_batch_normalization/beta':
                        'depthwise_conv.bn.bias',
                        'tpu_batch_normalization/gamma':
                        'depthwise_conv.bn.weight',
                        'tpu_batch_normalization/moving_mean':
                        'depthwise_conv.bn.running_mean',
                        'tpu_batch_normalization/moving_variance':
                        'depthwise_conv.bn.running_var',
                        'tpu_batch_normalization_1/beta':
                        'linear_conv.bn.bias',
                        'tpu_batch_normalization_1/gamma':
                        'linear_conv.bn.weight',
                        'tpu_batch_normalization_1/moving_mean':
                        'linear_conv.bn.running_mean',
                        'tpu_batch_normalization_1/moving_variance':
                        'linear_conv.bn.running_var',
                    }
                else:
                    mapping = {
                        'depthwise_conv2d/depthwise_kernel':
                        'depthwise_conv.conv.weight',
                        'conv2d/kernel':
                        'expand_conv.conv.weight',
                        'conv2d_1/kernel':
                        'linear_conv.conv.weight',
                        'tpu_batch_normalization/beta':
                        'expand_conv.bn.bias',
                        'tpu_batch_normalization/gamma':
                        'expand_conv.bn.weight',
                        'tpu_batch_normalization/moving_mean':
                        'expand_conv.bn.running_mean',
                        'tpu_batch_normalization/moving_variance':
                        'expand_conv.bn.running_var',
                        'tpu_batch_normalization_1/beta':
                        'depthwise_conv.bn.bias',
                        'tpu_batch_normalization_1/gamma':
                        'depthwise_conv.bn.weight',
                        'tpu_batch_normalization_1/moving_mean':
                        'depthwise_conv.bn.running_mean',
                        'tpu_batch_normalization_1/moving_variance':
                        'depthwise_conv.bn.running_var',
                        'tpu_batch_normalization_2/beta':
                        'linear_conv.bn.bias',
                        'tpu_batch_normalization_2/gamma':
                        'linear_conv.bn.weight',
                        'tpu_batch_normalization_2/moving_mean':
                        'linear_conv.bn.running_mean',
                        'tpu_batch_normalization_2/moving_variance':
                        'linear_conv.bn.running_var',
                    }
                mapping.update(base_mapping)
                suffix = mapping['/'.join(seg[2:])]
                m[prefix + '.' + suffix] = v
        elif seg[0] == 'resample_p6':
            prefix = 'neck.bifpn.0.p5_to_p6.0'
            mapping = {
                'conv2d/kernel': 'down_conv.weight',
                'conv2d/bias': 'down_conv.bias',
                'bn/beta': 'bn.bias',
                'bn/gamma': 'bn.weight',
                'bn/moving_mean': 'bn.running_mean',
                'bn/moving_variance': 'bn.running_var',
            }
            suffix = mapping['/'.join(seg[1:])]
            m[prefix + '.' + suffix] = v
        elif seg[0] == 'fpn_cells':
            fpn_idx = int(seg[1][5:])
            prefix = '.'.join(['neck', 'bifpn', str(fpn_idx)])
            fnode_id = int(seg[2][5])
            if fnode_id == 0:
                mapping = {
                    'op_after_combine5/conv/depthwise_kernel':
                    'conv6_up.depthwise_conv.weight',
                    'op_after_combine5/conv/pointwise_kernel':
                    'conv6_up.pointwise_conv.weight',
                    'op_after_combine5/conv/bias':
                    'conv6_up.pointwise_conv.bias',
                    'op_after_combine5/bn/beta':
                    'conv6_up.bn.bias',
                    'op_after_combine5/bn/gamma':
                    'conv6_up.bn.weight',
                    'op_after_combine5/bn/moving_mean':
                    'conv6_up.bn.running_mean',
                    'op_after_combine5/bn/moving_variance':
                    'conv6_up.bn.running_var',
                }
                if seg[3] != 'WSM' and seg[3] != 'WSM_1':
                    suffix = mapping['/'.join(seg[3:])]
                    if 'depthwise_conv' in suffix:
                        v = v.transpose(1, 0)
                    m[prefix + '.' + suffix] = v
                elif seg[3] == 'WSM':
                    p6_w1[fpn_idx][0] = v
                elif seg[3] == 'WSM_1':
                    p6_w1[fpn_idx][1] = v
                if torch.min(p6_w1[fpn_idx]) > -1e4:
                    m[prefix + '.p6_w1'] = p6_w1[fpn_idx]
            elif fnode_id == 1:
                base_mapping = {
                    'op_after_combine6/conv/depthwise_kernel':
                    'conv5_up.depthwise_conv.weight',
                    'op_after_combine6/conv/pointwise_kernel':
                    'conv5_up.pointwise_conv.weight',
                    'op_after_combine6/conv/bias':
                    'conv5_up.pointwise_conv.bias',
                    'op_after_combine6/bn/beta':
                    'conv5_up.bn.bias',
                    'op_after_combine6/bn/gamma':
                    'conv5_up.bn.weight',
                    'op_after_combine6/bn/moving_mean':
                    'conv5_up.bn.running_mean',
                    'op_after_combine6/bn/moving_variance':
                    'conv5_up.bn.running_var',
                }
                if fpn_idx == 0:
                    mapping = {
                        'resample_0_2_6/conv2d/kernel':
                        'p5_down_channel.down_conv.weight',
                        'resample_0_2_6/conv2d/bias':
                        'p5_down_channel.down_conv.bias',
                        'resample_0_2_6/bn/beta':
                        'p5_down_channel.bn.bias',
                        'resample_0_2_6/bn/gamma':
                        'p5_down_channel.bn.weight',
                        'resample_0_2_6/bn/moving_mean':
                        'p5_down_channel.bn.running_mean',
                        'resample_0_2_6/bn/moving_variance':
                        'p5_down_channel.bn.running_var',
                    }
                    base_mapping.update(mapping)
                if seg[3] != 'WSM' and seg[3] != 'WSM_1':
                    suffix = base_mapping['/'.join(seg[3:])]
                    if 'depthwise_conv' in suffix:
                        v = v.transpose(1, 0)
                    m[prefix + '.' + suffix] = v
                elif seg[3] == 'WSM':
                    p5_w1[fpn_idx][0] = v
                elif seg[3] == 'WSM_1':
                    p5_w1[fpn_idx][1] = v
                if torch.min(p5_w1[fpn_idx]) > -1e4:
                    m[prefix + '.p5_w1'] = p5_w1[fpn_idx]
            elif fnode_id == 2:
                base_mapping = {
                    'op_after_combine7/conv/depthwise_kernel':
                    'conv4_up.depthwise_conv.weight',
                    'op_after_combine7/conv/pointwise_kernel':
                    'conv4_up.pointwise_conv.weight',
                    'op_after_combine7/conv/bias':
                    'conv4_up.pointwise_conv.bias',
                    'op_after_combine7/bn/beta':
                    'conv4_up.bn.bias',
                    'op_after_combine7/bn/gamma':
                    'conv4_up.bn.weight',
                    'op_after_combine7/bn/moving_mean':
                    'conv4_up.bn.running_mean',
                    'op_after_combine7/bn/moving_variance':
                    'conv4_up.bn.running_var',
                }
                if fpn_idx == 0:
                    mapping = {
                        'resample_0_1_7/conv2d/kernel':
                        'p4_down_channel.down_conv.weight',
                        'resample_0_1_7/conv2d/bias':
                        'p4_down_channel.down_conv.bias',
                        'resample_0_1_7/bn/beta':
                        'p4_down_channel.bn.bias',
                        'resample_0_1_7/bn/gamma':
                        'p4_down_channel.bn.weight',
                        'resample_0_1_7/bn/moving_mean':
                        'p4_down_channel.bn.running_mean',
                        'resample_0_1_7/bn/moving_variance':
                        'p4_down_channel.bn.running_var',
                    }
                    base_mapping.update(mapping)
                if seg[3] != 'WSM' and seg[3] != 'WSM_1':
                    suffix = base_mapping['/'.join(seg[3:])]
                    if 'depthwise_conv' in suffix:
                        v = v.transpose(1, 0)
                    m[prefix + '.' + suffix] = v
                elif seg[3] == 'WSM':
                    p4_w1[fpn_idx][0] = v
                elif seg[3] == 'WSM_1':
                    p4_w1[fpn_idx][1] = v
                if torch.min(p4_w1[fpn_idx]) > -1e4:
                    m[prefix + '.p4_w1'] = p4_w1[fpn_idx]
            elif fnode_id == 3:

                base_mapping = {
                    'op_after_combine8/conv/depthwise_kernel':
                    'conv3_up.depthwise_conv.weight',
                    'op_after_combine8/conv/pointwise_kernel':
                    'conv3_up.pointwise_conv.weight',
                    'op_after_combine8/conv/bias':
                    'conv3_up.pointwise_conv.bias',
                    'op_after_combine8/bn/beta':
                    'conv3_up.bn.bias',
                    'op_after_combine8/bn/gamma':
                    'conv3_up.bn.weight',
                    'op_after_combine8/bn/moving_mean':
                    'conv3_up.bn.running_mean',
                    'op_after_combine8/bn/moving_variance':
                    'conv3_up.bn.running_var',
                }
                if fpn_idx == 0:
                    mapping = {
                        'resample_0_0_8/conv2d/kernel':
                        'p3_down_channel.down_conv.weight',
                        'resample_0_0_8/conv2d/bias':
                        'p3_down_channel.down_conv.bias',
                        'resample_0_0_8/bn/beta':
                        'p3_down_channel.bn.bias',
                        'resample_0_0_8/bn/gamma':
                        'p3_down_channel.bn.weight',
                        'resample_0_0_8/bn/moving_mean':
                        'p3_down_channel.bn.running_mean',
                        'resample_0_0_8/bn/moving_variance':
                        'p3_down_channel.bn.running_var',
                    }
                    base_mapping.update(mapping)
                if seg[3] != 'WSM' and seg[3] != 'WSM_1':
                    suffix = base_mapping['/'.join(seg[3:])]
                    if 'depthwise_conv' in suffix:
                        v = v.transpose(1, 0)
                    m[prefix + '.' + suffix] = v
                elif seg[3] == 'WSM':
                    p3_w1[fpn_idx][0] = v
                elif seg[3] == 'WSM_1':
                    p3_w1[fpn_idx][1] = v
                if torch.min(p3_w1[fpn_idx]) > -1e4:
                    m[prefix + '.p3_w1'] = p3_w1[fpn_idx]
            elif fnode_id == 4:
                base_mapping = {
                    'op_after_combine9/conv/depthwise_kernel':
                    'conv4_down.depthwise_conv.weight',
                    'op_after_combine9/conv/pointwise_kernel':
                    'conv4_down.pointwise_conv.weight',
                    'op_after_combine9/conv/bias':
                    'conv4_down.pointwise_conv.bias',
                    'op_after_combine9/bn/beta':
                    'conv4_down.bn.bias',
                    'op_after_combine9/bn/gamma':
                    'conv4_down.bn.weight',
                    'op_after_combine9/bn/moving_mean':
                    'conv4_down.bn.running_mean',
                    'op_after_combine9/bn/moving_variance':
                    'conv4_down.bn.running_var',
                }
                if fpn_idx == 0:
                    mapping = {
                        'resample_0_1_9/conv2d/kernel':
                        'p4_level_connection.down_conv.weight',
                        'resample_0_1_9/conv2d/bias':
                        'p4_level_connection.down_conv.bias',
                        'resample_0_1_9/bn/beta':
                        'p4_level_connection.bn.bias',
                        'resample_0_1_9/bn/gamma':
                        'p4_level_connection.bn.weight',
                        'resample_0_1_9/bn/moving_mean':
                        'p4_level_connection.bn.running_mean',
                        'resample_0_1_9/bn/moving_variance':
                        'p4_level_connection.bn.running_var',
                    }
                    base_mapping.update(mapping)
                if seg[3] != 'WSM' and seg[3] != 'WSM_1' and seg[3] != 'WSM_2':
                    suffix = base_mapping['/'.join(seg[3:])]
                    if 'depthwise_conv' in suffix:
                        v = v.transpose(1, 0)
                    m[prefix + '.' + suffix] = v
                elif seg[3] == 'WSM':
                    p4_w2[fpn_idx][0] = v
                elif seg[3] == 'WSM_1':
                    p4_w2[fpn_idx][1] = v
                elif seg[3] == 'WSM_2':
                    p4_w2[fpn_idx][2] = v
                if torch.min(p4_w2[fpn_idx]) > -1e4:
                    m[prefix + '.p4_w2'] = p4_w2[fpn_idx]
            elif fnode_id == 5:
                base_mapping = {
                    'op_after_combine10/conv/depthwise_kernel':
                    'conv5_down.depthwise_conv.weight',
                    'op_after_combine10/conv/pointwise_kernel':
                    'conv5_down.pointwise_conv.weight',
                    'op_after_combine10/conv/bias':
                    'conv5_down.pointwise_conv.bias',
                    'op_after_combine10/bn/beta':
                    'conv5_down.bn.bias',
                    'op_after_combine10/bn/gamma':
                    'conv5_down.bn.weight',
                    'op_after_combine10/bn/moving_mean':
                    'conv5_down.bn.running_mean',
                    'op_after_combine10/bn/moving_variance':
                    'conv5_down.bn.running_var',
                }
                if fpn_idx == 0:
                    mapping = {
                        'resample_0_2_10/conv2d/kernel':
                        'p5_level_connection.down_conv.weight',
                        'resample_0_2_10/conv2d/bias':
                        'p5_level_connection.down_conv.bias',
                        'resample_0_2_10/bn/beta':
                        'p5_level_connection.bn.bias',
                        'resample_0_2_10/bn/gamma':
                        'p5_level_connection.bn.weight',
                        'resample_0_2_10/bn/moving_mean':
                        'p5_level_connection.bn.running_mean',
                        'resample_0_2_10/bn/moving_variance':
                        'p5_level_connection.bn.running_var',
                    }
                    base_mapping.update(mapping)
                if seg[3] != 'WSM' and seg[3] != 'WSM_1' and seg[3] != 'WSM_2':
                    suffix = base_mapping['/'.join(seg[3:])]
                    if 'depthwise_conv' in suffix:
                        v = v.transpose(1, 0)
                    m[prefix + '.' + suffix] = v
                elif seg[3] == 'WSM':
                    p5_w2[fpn_idx][0] = v
                elif seg[3] == 'WSM_1':
                    p5_w2[fpn_idx][1] = v
                elif seg[3] == 'WSM_2':
                    p5_w2[fpn_idx][2] = v
                if torch.min(p5_w2[fpn_idx]) > -1e4:
                    m[prefix + '.p5_w2'] = p5_w2[fpn_idx]
            elif fnode_id == 6:
                base_mapping = {
                    'op_after_combine11/conv/depthwise_kernel':
                    'conv6_down.depthwise_conv.weight',
                    'op_after_combine11/conv/pointwise_kernel':
                    'conv6_down.pointwise_conv.weight',
                    'op_after_combine11/conv/bias':
                    'conv6_down.pointwise_conv.bias',
                    'op_after_combine11/bn/beta':
                    'conv6_down.bn.bias',
                    'op_after_combine11/bn/gamma':
                    'conv6_down.bn.weight',
                    'op_after_combine11/bn/moving_mean':
                    'conv6_down.bn.running_mean',
                    'op_after_combine11/bn/moving_variance':
                    'conv6_down.bn.running_var',
                }
                if seg[3] != 'WSM' and seg[3] != 'WSM_1' and seg[3] != 'WSM_2':
                    suffix = base_mapping['/'.join(seg[3:])]
                    if 'depthwise_conv' in suffix:
                        v = v.transpose(1, 0)
                    m[prefix + '.' + suffix] = v
                elif seg[3] == 'WSM':
                    p6_w2[fpn_idx][0] = v
                elif seg[3] == 'WSM_1':
                    p6_w2[fpn_idx][1] = v
                elif seg[3] == 'WSM_2':
                    p6_w2[fpn_idx][2] = v
                if torch.min(p6_w2[fpn_idx]) > -1e4:
                    m[prefix + '.p6_w2'] = p6_w2[fpn_idx]
            elif fnode_id == 7:
                base_mapping = {
                    'op_after_combine12/conv/depthwise_kernel':
                    'conv7_down.depthwise_conv.weight',
                    'op_after_combine12/conv/pointwise_kernel':
                    'conv7_down.pointwise_conv.weight',
                    'op_after_combine12/conv/bias':
                    'conv7_down.pointwise_conv.bias',
                    'op_after_combine12/bn/beta':
                    'conv7_down.bn.bias',
                    'op_after_combine12/bn/gamma':
                    'conv7_down.bn.weight',
                    'op_after_combine12/bn/moving_mean':
                    'conv7_down.bn.running_mean',
                    'op_after_combine12/bn/moving_variance':
                    'conv7_down.bn.running_var',
                }
                if seg[3] != 'WSM' and seg[3] != 'WSM_1' and seg[3] != 'WSM_2':
                    suffix = base_mapping['/'.join(seg[3:])]
                    if 'depthwise_conv' in suffix:
                        v = v.transpose(1, 0)
                    m[prefix + '.' + suffix] = v
                elif seg[3] == 'WSM':
                    p7_w2[fpn_idx][0] = v
                elif seg[3] == 'WSM_1':
                    p7_w2[fpn_idx][1] = v
                if torch.min(p7_w2[fpn_idx]) > -1e4:
                    m[prefix + '.p7_w2'] = p7_w2[fpn_idx]
        elif seg[0] == 'box_net':
            if 'box-predict' in seg[1]:
                prefix = '.'.join(['bbox_head', 'reg_header'])
                base_mapping = {
                    'depthwise_kernel': 'depthwise_conv.weight',
                    'pointwise_kernel': 'pointwise_conv.weight',
                    'bias': 'pointwise_conv.bias'
                }
                suffix = base_mapping['/'.join(seg[2:])]
                if 'depthwise_conv' in suffix:
                    v = v.transpose(1, 0)
                m[prefix + '.' + suffix] = v
            elif 'bn' in seg[1]:
                bbox_conv_idx = int(seg[1][4])
                bbox_bn_idx = int(seg[1][9]) - 3
                prefix = '.'.join([
                    'bbox_head', 'reg_bn_list',
                    str(bbox_conv_idx),
                    str(bbox_bn_idx)
                ])
                base_mapping = {
                    'beta': 'bias',
                    'gamma': 'weight',
                    'moving_mean': 'running_mean',
                    'moving_variance': 'running_var'
                }
                suffix = base_mapping['/'.join(seg[2:])]
                m[prefix + '.' + suffix] = v
            else:
                bbox_conv_idx = int(seg[1][4])
                prefix = '.'.join(
                    ['bbox_head', 'reg_conv_list',
                     str(bbox_conv_idx)])
                base_mapping = {
                    'depthwise_kernel': 'depthwise_conv.weight',
                    'pointwise_kernel': 'pointwise_conv.weight',
                    'bias': 'pointwise_conv.bias'
                }
                suffix = base_mapping['/'.join(seg[2:])]
                if 'depthwise_conv' in suffix:
                    v = v.transpose(1, 0)
                m[prefix + '.' + suffix] = v
        elif seg[0] == 'class_net':
            if 'class-predict' in seg[1]:
                prefix = '.'.join(['bbox_head', 'cls_header'])
                base_mapping = {
                    'depthwise_kernel': 'depthwise_conv.weight',
                    'pointwise_kernel': 'pointwise_conv.weight',
                    'bias': 'pointwise_conv.bias'
                }
                suffix = base_mapping['/'.join(seg[2:])]
                if 'depthwise_conv' in suffix:
                    v = v.transpose(1, 0)
                m[prefix + '.' + suffix] = v
            elif 'bn' in seg[1]:
                cls_conv_idx = int(seg[1][6])
                cls_bn_idx = int(seg[1][11]) - 3
                prefix = '.'.join([
                    'bbox_head', 'cls_bn_list',
                    str(cls_conv_idx),
                    str(cls_bn_idx)
                ])
                base_mapping = {
                    'beta': 'bias',
                    'gamma': 'weight',
                    'moving_mean': 'running_mean',
                    'moving_variance': 'running_var'
                }
                suffix = base_mapping['/'.join(seg[2:])]
                m[prefix + '.' + suffix] = v
            else:
                cls_conv_idx = int(seg[1][6])
                prefix = '.'.join(
                    ['bbox_head', 'cls_conv_list',
                     str(cls_conv_idx)])
                base_mapping = {
                    'depthwise_kernel': 'depthwise_conv.weight',
                    'pointwise_kernel': 'pointwise_conv.weight',
                    'bias': 'pointwise_conv.bias'
                }
                suffix = base_mapping['/'.join(seg[2:])]
                if 'depthwise_conv' in suffix:
                    v = v.transpose(1, 0)
                m[prefix + '.' + suffix] = v
    return m


def parse_args():
    parser = argparse.ArgumentParser(
        description='convert efficientdet weight from tensorflow to pytorch')
    parser.add_argument(
        '--backbone',
        type=str,
        help='efficientnet model name, like efficientnet-b0')
    parser.add_argument(
        '--tensorflow_weight',
        type=str,
        help='efficientdet tensorflow weight name, like efficientdet-d0/model')
    parser.add_argument(
        '--out_weight',
        type=str,
        help='efficientdet pytorch weight name like demo.pth')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model_name = args.backbone
    ori_weight_name = args.tensorflow_weight
    out_name = args.out_weight

    repeat_map = {
        0: 3,
        1: 4,
        2: 5,
        3: 6,
        4: 7,
        5: 7,
        6: 8,
        7: 8,
    }

    reader = py_checkpoint_reader.NewCheckpointReader(ori_weight_name)
    weights = {
        n: torch.as_tensor(tf2pth(reader.get_tensor(n)))
        for (n, _) in reader.get_variable_to_shape_map().items()
    }
    bifpn_repeats = repeat_map[int(model_name[14])]
    out = convert_key(model_name, bifpn_repeats, weights)
    result = {'state_dict': out}
    torch.save(result, out_name)


if __name__ == '__main__':
    main()
