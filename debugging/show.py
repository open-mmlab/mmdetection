"""Show.

Script to show using imshow_det_bboxes

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    December 03, 2019
"""
import numpy as np
from PIL import Image, ImageDraw
from debugging.classes_lookup import DEEPSCORES
import torch
from pycocotools.coco import COCO
from tqdm import tqdm
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


def cati(coco: COCO, output_fp: str, pred: bool, img_ids: list = None,
         img_dir = None):
    """COCO Anns to Image.

    Overlays bounding boxes with categories from a COCO ann_list onto the image
    specified in img_fp.

    Args:
        coco: The coco API object with the desired annotations.
        output_fp: Where to output the files to.
        pred: True if this is the prediction, false if ground truth.
        img_ids: Image IDs to output
        img_dir str or None: The directory where the images are located.
    """
    if not img_dir:
        img_dir ='/workspace/deep_scores_dense/images_png/'

    labels = ('', 'bbox-only_')
    color = (255, 0, 0) if pred else (0, 255, 0)

    if img_ids is None:
        img_ids = [1]

    for img_id in tqdm(img_ids):
        ims = [Image.open(img_dir + coco.imgs[img_id]['file_name']),
               Image.new('RGB',
                         Image.open(
                             img_dir + coco.imgs[img_id]['file_name']).size
                         )
               ]

        for label, im in zip(labels, ims):
            draw = ImageDraw.Draw(im)
            ann_ids = coco.getAnnIds(imgIds=[img_id])
            for ann_id in ann_ids:
                ann = coco.anns[ann_id]
                bbox = ann['bbox']
                # Draw bounding box rectangle
                draw.rectangle((bbox[0], bbox[1],
                                bbox[0] + bbox[2],
                                bbox[1] + bbox[3]),
                               outline=color)

                # Draw class label
                class_label = coco.cats[ann['category_id']]['name']
                text_size = draw.textsize(class_label)  # Returns (w, h)
                if label == '':
                    draw.rectangle((bbox[0],
                                    bbox[1] - text_size[1] - 4,
                                    bbox[0] + text_size[0] + 4,
                                    bbox[1]),
                                   fill=(53, 103, 184))  # Nice navy blue
                else:
                    draw.rectangle((bbox[0],
                                    bbox[1] - text_size[1] - 4,
                                    bbox[0] + text_size[0] + 4,
                                    bbox[1]),
                                   fill=color)

                draw.text((bbox[0] + 2, bbox[1] - text_size[1] - 2),
                          class_label,
                          fill=(255, 255, 255))
            im.save(output_fp + '{}{}.png'.format(label, img_id))


def ttp(img: torch.Tensor, save_path: str = None) -> Image:
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


def compare(gt_file: str, pred_file: str, diff_file: str, img_ids: list):
    """Compares two images."""
    for img_id in tqdm(img_ids):
        gt_array = np.array(Image.open(gt_file + 'bbox-only_' + str(img_id)
                                       + '.png'))
        pred_array = np.array(Image.open(pred_file + 'bbox-only_'
                                         + str(img_id) + '.png'))

        assert gt_array.shape == pred_array.shape, "Somehow the outputs " \
                                                   "aren't the same size?"

        diff_array = gt_array + pred_array

        Image.fromarray(diff_array).save(diff_file + str(img_id) + '.png')


def from_file(results: str, coco: str, img_ind: int or list, img_path: str):
    """Does cati from a file."""
    if not img_ind:
        img_ind = [0]
    coco = COCO(coco)
    results = coco.loadRes(results)

    keys = list(coco.imgs.keys())

    if len(img_ind) > 0:
        img_id = [coco.imgs[keys[ind]]['id'] for ind in img_ind]
    else:
        img_id = [coco.imgs[key]['id'] for key in keys]

    gt_path = '/workspace/outputs/gt-'
    pred_path = '/workspace/outputs/pred-'
    diff_path = '/workspace/outputs/diff-'

    print('Writing ground truth bboxes')
    cati(coco, gt_path, False, img_id, img_path)
    print('Writing predict bboxes')
    cati(results, pred_path, True, img_id, img_path)

    print('Writing differences...')
    compare(gt_path, pred_path, diff_path, img_id)

    print('Done!')


def parse_arguments():
    desc = "shows results of training."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('RESULTS', type=str,
                        help='results file')
    parser.add_argument('COCO', type=str,
                        help='coco annotations file')
    parser.add_argument('-i', '--IMG-PATH', type=str, nargs='?',
                        help='path to the image directory')
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
    elif args.save is not None:
        from_file(args.RESULTS, args.COCO, args.save, args.IMG_PATH)
    else:
        raise ValueError("At least one mode option must be selected.")
