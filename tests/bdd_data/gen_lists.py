import argparse
import os
from os import path as osp
import sys


def gen_list(data_root, data_dir, list_dir, phase, list_type, suffix='.jpg'):
    phase_dir = osp.join(data_root, data_dir, phase)
    if not osp.exists(phase_dir):
        raise ValueError('Can not find folder {}'.format(phase_dir))
    images = sorted([osp.join(data_dir, phase, n)
                     for n in os.listdir(phase_dir)
                     if n[-len(suffix):] == suffix])
    print('Found', len(images), 'items in', data_dir, phase)
    out_path = osp.join(list_dir, '{}_{}.txt'.format(phase, list_type))
    if not osp.exists(list_dir):
        os.makedirs(list_dir)
    print('Writing', out_path)
    with open(out_path, 'w') as fp:
        fp.write('\n'.join(images))


def gen_images(data_root, list_dir, image_type='100k'):
    for phase in ['train', 'val', 'test']:
        gen_list(data_root, osp.join('images', image_type),
                 list_dir, phase, 'images', '.jpg')


def gen_drivable(data_root):
    image_type = '100k'
    label_dir = 'drivable_maps/labels'
    list_dir = 'lists/100k/drivable'

    gen_images(data_root, list_dir, image_type)

    for p in ['train', 'val']:
        gen_list(data_root, label_dir, list_dir, p, 'labels',
                 'drivable_id.png')


def gen_seg(data_root):
    image_type = '10k'
    label_dir = 'seg_maps/labels'
    list_dir = 'lists/10k/seg'

    gen_images(data_root, list_dir, image_type)

    for p in ['train', 'val']:
        gen_list(data_root, label_dir, list_dir, p, 'labels',
                 'train_id.png')


if __name__ == '__main__':
    gen_drivable(sys.argv[1])
    gen_seg(sys.argv[1])
