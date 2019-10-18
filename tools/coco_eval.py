from argparse import ArgumentParser

from mmdet.core import coco_eval


def main():
    parser = ArgumentParser(description='COCO Evaluation')
    parser.add_argument('result', help='result file path')
    parser.add_argument('--ann', help='annotation file path')
    parser.add_argument(
        '--types',
        type=str,
        nargs='+',
        choices=['proposal_fast', 'proposal', 'bbox', 'segm', 'keypoint'],
        default=['bbox'],
        help='result types')
    parser.add_argument(
        '--max-dets',
        type=int,
        nargs='+',
        default=[100, 300, 1000],
        help='proposal numbers, only used for recall evaluation')
    parser.add_argument(
        '--classwise', action='store_true', help='whether eval class wise ap')
    args = parser.parse_args()
    coco_eval(args.result, args.types, args.ann, args.max_dets, args.classwise)


if __name__ == '__main__':
    main()
