import argparse

import torch

from mmdet.models.utils.ckpt_convert import swin_converter


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in official pretrained swin models to'
        'MMDetection style.')
    parser.add_argument('src', help='src detection model path')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = torch.load(args.src, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    weight = swin_converter(state_dict,prefix='backbone.')

    if 'state_dict' in checkpoint:
        checkpoint['state_dict'] = weight
    elif 'model' in checkpoint:
        checkpoint['model'] = weight
    else:
        checkpoint = weight
    with open(args.dst, 'wb') as f:
        torch.save(checkpoint, f)


if __name__ == '__main__':
    main()