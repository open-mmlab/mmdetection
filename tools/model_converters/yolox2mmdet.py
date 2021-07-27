import argparse
from collections import OrderedDict
import torch


def convert(src, dst):
    data = torch.load(src)['model']
    new_state_dict = OrderedDict()
    for k, v in data.items():
        if 'backbone.backbone' in k:
            k = k.replace('backbone.backbone', 'backbone')
            for i in range(1, 5):
                if f'dark{i+1}' in k:
                    k = k.replace(f'dark{i+1}', f'stage{i}')
                    if '.m.' in k:
                        k = k.replace('.m.', '.blocks.')
                    else:
                        if 'stage4.1' in k:
                            continue
                        k = k.replace('conv1', 'main_conv')
                        k = k.replace('conv2', 'short_conv')
                        k = k.replace('conv3', 'final_conv')
                    break
        elif 'backbone' in k:
            k = k.replace('backbone', 'neck')
            k = k.replace('lateral_conv', 'reduce_layers.')
            k = k.replace('reduce_conv', 'reduce_layers.')
            k = k.replace('bu_conv2', 'downsamples.0')
            k = k.replace('bu_conv1', 'downsamples.1')
            if 'C3' in k:
                k = k.replace('C3_p4', 'top_down_blocks.0')
                k = k.replace('C3_p3', 'top_down_blocks.1')
                k = k.replace('C3_n3', 'bottom_up_blocks.0')
                k = k.replace('C3_n4', 'bottom_up_blocks.1')
                if '.m.' in k:
                    k = k.replace('.m.', '.blocks.')
                else:
                    k = k.replace('conv1', 'main_conv')
                    k = k.replace('conv2', 'short_conv')
                    k = k.replace('conv3', 'final_conv')
        if 'head.stems' in k:
            k = k.replace('head.stems', 'neck.out_convs')

        elif 'head' in k:
            k = k.replace('head', 'bbox_head')
            k = k.replace('cls_convs', 'multi_level_cls_convs')
            k = k.replace('reg_convs', 'multi_level_reg_convs')
            k = k.replace('cls_preds', 'multi_level_conv_cls')
            k = k.replace('reg_preds', 'multi_level_conv_reg')
            k = k.replace('obj_preds', 'multi_level_conv_obj')
        new_state_dict[k] = v

    data = {"state_dict": new_state_dict}
    torch.save(data, dst)


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src detectron model path')
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()
