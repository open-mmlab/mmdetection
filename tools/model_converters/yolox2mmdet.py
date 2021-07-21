import argparse
from collections import OrderedDict
import torch


def convert(src, dst):
    data = torch.load(src)['model']
    new_state_dict = OrderedDict()
    for k, v in data.items():
        if 'head' in k:
            k = k.replace('head', 'bbox_head')
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
