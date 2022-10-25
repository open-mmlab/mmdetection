# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from collections import OrderedDict

import torch
from mmengine.fileio import load
from mmengine.runner import save_checkpoint


def convert(src: str, dst: str, prefix: str = 'd2_model') -> None:
    """Convert Detectron2 checkpoint to MMDetection style.

    Args:
        src (str): The Detectron2 checkpoint path, should endswith `pkl`.
        dst (str): The MMDetection checkpoint path.
        prefix (str): The prefix of MMDetection model, defaults to 'd2_model'.
    """
    # load arch_settings
    assert src.endswith('pkl'), \
        'the source Detectron2 checkpoint should endswith `pkl`.'
    d2_model = load(src, encoding='latin1').get('model')
    assert d2_model is not None

    # convert to mmdet style
    dst_state_dict = OrderedDict()
    for name, value in d2_model.items():
        if not isinstance(value, torch.Tensor):
            value = torch.from_numpy(value)
        dst_state_dict[f'{prefix}.{name}'] = value

    mmdet_model = dict(state_dict=dst_state_dict, meta=dict())
    save_checkpoint(mmdet_model, dst)
    print(f'Convert Detectron2 model {src} to MMDetection model {dst}')


def main():
    parser = argparse.ArgumentParser(
        description='Convert Detectron2 checkpoint to MMDetection style')
    parser.add_argument('src', help='Detectron2 model path')
    parser.add_argument('dst', help='MMDetectron model save path')
    parser.add_argument(
        '--prefix', default='d2_model', type=str, help='prefix of the model')
    args = parser.parse_args()
    convert(args.src, args.dst, args.prefix)


if __name__ == '__main__':
    main()
