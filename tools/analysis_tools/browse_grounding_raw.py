# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os.path as osp

import cv2
import numpy as np
from mmcv.image import imfrombytes, imwrite
from mmengine.fileio import get
from mmengine.structures import InstanceData
from mmengine.utils import mkdir_or_exist

from mmdet.structures import DetDataSample
from mmdet.visualization import DetLocalVisualizer
from mmdet.visualization.palette import _get_adaptive_scales

# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('data_root')
    parser.add_argument('ann_file')
    parser.add_argument('img_prefix')
    parser.add_argument('--label-map-file', '-m', default=None)
    parser.add_argument(
        '--output-dir',
        '-o',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument('--show-num', '-n', type=int, default=30)
    parser.add_argument('--shuffle', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=0,
        help='the interval of show (s)')
    args = parser.parse_args()
    return args


def draw_all_character(visualizer, characters, w):
    start_index = 2
    y_index = 5
    for char in characters:
        if isinstance(char, str):
            visualizer.draw_texts(
                str(char),
                positions=np.array([start_index, y_index]),
                colors=(0, 0, 0),
                font_families='monospace')
            start_index += len(char) * 8
        else:
            visualizer.draw_texts(
                str(char[0]),
                positions=np.array([start_index, y_index]),
                colors=char[1],
                font_families='monospace')
            start_index += len(char[0]) * 8

        if start_index > w - 10:
            start_index = 2
            y_index += 15

    drawn_text = visualizer.get_image()
    return drawn_text


def main():
    args = parse_args()
    assert args.show_num > 0

    local_path = osp.join(args.data_root, args.ann_file)
    with open(local_path, 'r') as f:
        data_list = [json.loads(line) for line in f]

    dataset_index = list(range(len(data_list)))
    if args.shuffle:
        import random
        random.shuffle(dataset_index)

    if args.label_map_file is not None:
        label_map_file = osp.join(args.data_root, args.label_map_file)
        with open(label_map_file, 'r') as file:
            label_map = json.load(file)

    visualizer = DetLocalVisualizer()

    for i in dataset_index[:args.show_num]:
        item = data_list[i]

        img_path = osp.join(args.data_root, args.img_prefix, item['filename'])
        if backend_args is not None:
            img_bytes = get(img_path, backend_args)
            img = imfrombytes(img_bytes, flag='color')
        else:
            img = cv2.imread(img_path)
        img = img[..., [2, 1, 0]]  # bgr to rgb

        base_name, extension = osp.splitext(item['filename'])

        out_file = osp.join(args.output_dir, base_name + '_' + str(i) +
                            extension) if args.output_dir is not None else None

        if args.output_dir is not None:
            mkdir_or_exist(args.output_dir)

        if 'detection' in item:
            anno = item['detection']

            instances = [obj for obj in anno['instances']]
            bboxes = [obj['bbox'] for obj in instances]
            bbox_labels = [int(obj['label']) for obj in instances]
            label_names = [label_map[str(label)] for label in bbox_labels]

            data_sample = DetDataSample()
            gt_instances = InstanceData()
            if len(instances) > 0 and 'score' in instances[0]:
                score = [obj['score'] for obj in instances]
                gt_instances['scores'] = np.array(score)

            gt_instances['bboxes'] = np.array(bboxes).reshape(-1, 4)
            gt_instances['labels'] = np.array(bbox_labels)
            gt_instances['label_names'] = label_names
            data_sample.gt_instances = gt_instances

            visualizer.add_datasample(
                osp.basename(img_path),
                img,
                data_sample,
                draw_pred=False,
                show=not args.not_show,
                wait_time=args.show_interval,
                out_file=out_file)
        elif 'grounding' in item:
            anno = item['grounding']
            text = anno['caption']
            regions = anno['regions']

            max_label = len(regions) if len(regions) > 0 else 0
            palette = np.random.randint(0, 256, size=(max_label + 1, 3))
            bbox_palette = [tuple(c) for c in palette]
            # bbox_palette = get_palette('random', max_label + 1)
            colors = [bbox_palette[label] for label in range(max_label)]

            visualizer.set_image(img)

            gt_tokens_positive = []
            for i, region in enumerate(regions):
                bbox = region['bbox']
                bbox = np.array(bbox).reshape(-1, 4)
                tokens_positive = region['tokens_positive']
                gt_tokens_positive.append(tokens_positive)
                visualizer.draw_bboxes(
                    bbox,
                    edge_colors=colors[i],
                    face_colors=colors[i],
                    alpha=0.3)
                visualizer.draw_bboxes(bbox, edge_colors=colors[i], alpha=1)

                if 'score' in region:
                    areas = (bbox[:, 3] - bbox[:, 1]) * (
                        bbox[:, 2] - bbox[:, 0])
                    scales = _get_adaptive_scales(areas)
                    score = region['score'][0]
                    score = [str(s) for s in score]
                    font_sizes = [
                        int(13 * scales[i]) for i in range(len(scales))
                    ]
                    visualizer.draw_texts(
                        score,
                        bbox[:, :2].astype(np.int32),
                        colors=(255, 255, 255),
                        font_sizes=font_sizes,
                        bboxes=[{
                            'facecolor': 'black',
                            'alpha': 0.8,
                            'pad': 0.7,
                            'edgecolor': 'none'
                        }] * len(bbox))

            drawn_img = visualizer.get_image()
            new_image = np.ones((100, img.shape[1], 3), dtype=np.uint8) * 255
            visualizer.set_image(new_image)

            split_by_character = [char for char in text]
            characters = []
            start_index = 0
            end_index = 0
            for w in split_by_character:
                end_index += len(w)
                is_find = False
                for i, positive in enumerate(gt_tokens_positive):
                    for p in positive:
                        if start_index >= p[0] and end_index <= p[1]:
                            characters.append([w, colors[i]])
                            is_find = True
                            break
                    if is_find:
                        break
                if not is_find:
                    characters.append([w, (0, 0, 0)])
                start_index = end_index

            drawn_text = draw_all_character(visualizer, characters,
                                            img.shape[1])
            drawn_img = np.concatenate((drawn_img, drawn_text), axis=0)

            if not args.not_show:
                visualizer.show(
                    drawn_img,
                    win_name=base_name,
                    wait_time=args.show_interval)

            if out_file is not None:
                imwrite(drawn_img[..., ::-1], out_file)

        elif 'referring' in item:
            referring = item['referring']

            max_label = len(referring) if len(referring) > 0 else 0
            palette = np.random.randint(0, 256, size=(max_label + 1, 3))
            bbox_palette = [tuple(c) for c in palette]
            # bbox_palette = get_palette('random', max_label + 1)
            colors = [bbox_palette[label] for label in range(max_label)]

            visualizer.set_image(img)
            phrases = []
            for i, ref in enumerate(referring):
                bbox = ref['bbox']
                phrase = ref['phrase']
                phrases.append(' // '.join(phrase))
                bbox = np.array(bbox).reshape(-1, 4)

                visualizer.draw_bboxes(
                    bbox,
                    edge_colors=colors[i],
                    face_colors=colors[i],
                    alpha=0.3)
                visualizer.draw_bboxes(bbox, edge_colors=colors[i], alpha=1)
            drawn_img = visualizer.get_image()

            new_image = np.ones((100, img.shape[1], 3), dtype=np.uint8) * 255
            visualizer.set_image(new_image)

            start_index = 2
            y_index = 5

            chunk_size = max(min(img.shape[1] - 400, 70), 50)
            for i, p in enumerate(phrases):
                chunk_p = [
                    p[i:i + chunk_size] for i in range(0, len(p), chunk_size)
                ]
                for cp in chunk_p:
                    visualizer.draw_texts(
                        cp,
                        positions=np.array([start_index, y_index]),
                        colors=colors[i],
                        font_families='monospace')
                    y_index += 15

            drawn_text = visualizer.get_image()
            drawn_img = np.concatenate((drawn_img, drawn_text), axis=0)

            if not args.not_show:
                visualizer.show(
                    drawn_img,
                    win_name=base_name,
                    wait_time=args.show_interval)

            if out_file is not None:
                imwrite(drawn_img[..., ::-1], out_file)


if __name__ == '__main__':
    main()
