"""Show.

Script to show using imshow_det_bboxes

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    December 03, 2019
"""
import numpy as np
from mmcv.visualization.image import imshow_det_bboxes
from PIL import Image, ImageDraw
from debugging.classes_lookup import DEEPSCORES
import torch
from pycocotools.coco import COCO
from tqdm import tqdm
import sys
from mmdet.core.evaluation.coco_utils import coco_eval
import argparse


def rfd(data, result, file_suffix, threshold=0.17, show=False):
    """Run this show script during debugging and use it with data_out.

    Notes:
        rfd stands for run from debug.
    """
    r_list = []
    labels = []

    for l, r in enumerate(result):
        if r.size != 0:
            for i in range(r.shape[0]):
                r_list.append(r[i])
                labels.append(l)

    r_array = np.array(r_list)
    labels = np.array(labels)
    # img = data['img'][0]
    # img = np.rollaxis(img.numpy()[0], 0, 3).astype('uint8')
    img = ttp(data['img'][0][0])
    draw = ImageDraw.Draw(img)
    for label, bbox in zip(labels, r_list):
        if bbox[4] > threshold:
            draw.rectangle(bbox[0:4], fill=None, outline=(0, 255, 0))
            draw.text((bbox[0], bbox[1] - 10), DEEPSCORES[label],
                      fill=(255, 255, 0))
    img.save('/workspace/img{}.png'.format(file_suffix))

    if show:
        img.show()

    return r_array


def cati(coco:COCO, output_fp: str, img_ids: list=None,
         img_dir: str='/workspace/deep_scores_dense/images_png/'):
    """COCO Anns to Image.

    Overlays bounding boxes with categories from a COCO ann_list onto the image
    specified in img_fp.

    Args:
        coco: The coco API object with the desired annotations
    """
    out_list = []
    if img_ids is None:
        img_ids=[1]

    for img_id in tqdm(img_ids):
        im = Image.open(img_dir + coco.imgs[img_id]['file_name'])
        draw = ImageDraw.Draw(im)
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        for ann_id in ann_ids:
            ann = coco.anns[ann_id]
            bbox = ann['bbox']
            # Draw bounding box rectangle
            draw.rectangle((bbox[0], bbox[1],
                            bbox[0] + bbox[2],
                            bbox[1] + bbox[3]),
                           outline=(0, 255, 0))

            # Draw class label
            class_label = coco.cats[ann['category_id']]['name']
            text_size = draw.textsize(class_label)  # Returns (w, h)
            draw.rectangle((bbox[0],
                            bbox[1] - text_size[1] - 4,
                            bbox[0] + text_size[0] + 4,
                            bbox[1]),
                           fill=(46, 92, 166),  # Nice navy blue
                           outline=(255, 255, 255))
            draw.text((bbox[0] + 2, bbox[1] - text_size[1] - 2),
                      class_label,
                      fill=(255, 255, 255))
        im.save(output_fp + 'img{}.png'.format(img_id))
        out_list.append(im)


def ttp(img: torch.Tensor, save_path: str=None) -> Image:
    """Tensor to PIL image. Saves to a path if specified.

    Shape:
        img: (3, h, w)
    """
    if img.device != torch.device('cpu'):
        img = img.detach().cpu()
    img = Image.fromarray(np.rollaxis(img.numpy(),
                                      0, 3).astype('uint8'))
    if save_path is not None:
        img.save(save_path)

    return img


def from_file(results: str, coco: str, img_ind: int or list):
    """Does cati from a file."""
    if not img_ind:
        img_ind = [0]
    coco = COCO(coco)
    results = coco.loadRes(results)

    keys = list(coco.imgs.keys())

    img_id = [coco.imgs[keys[ind]]['id'] for ind in img_ind]

    print('Writing ground truth bboxes')
    cati(coco, '/workspace/outputs/gt-', img_id)
    print('Writing predict bboxes')
    cati(results, '/workspace/outputs/pred-', img_id)
    print('Done!')


def parse_arguments():
    desc = "shows results of training."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('RESULTS', type=str,
                        help='results file')
    parser.add_argument('COCO', type=str,
                        help='coco annotations file')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('-e', '--eval', action='store_true',
                      help='evaluate AP and AR')
    mode.add_argument('-c', '--class-wise-eval', action='store_true',
                      help='evaluate classwise AP')
    mode.add_argument('-s', '--save', type=int, nargs='*',
                      help='save')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    if args.eval:
        coco_eval(args.RESULTS, ['bbox'], args.COCO, max_dets=300)
    elif args.class_wise_eval:
        coco_eval(args.RESULTS, ['bbox'], args.COCO, max_dets=300,
        classwise=True)
    else:
       from_file(args.RESULTS, args.COCO, args.save)
