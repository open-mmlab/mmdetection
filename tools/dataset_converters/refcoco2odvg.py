import argparse
import os.path as osp

import jsonlines
from pycocotools.coco import COCO
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='refcoco to odvg')
    parser.add_argument('mdetr_anno_dir', type=str)
    parser.add_argument('--out-dir', '-o', type=str)
    args = parser.parse_args()
    return args


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    return True


def process_item(args, filename):
    path = osp.join(args.mdetr_anno_dir, filename)
    coco = COCO(path)

    ids = list(sorted(coco.imgs.keys()))

    out_results = []
    for img_id in tqdm(ids):
        if isinstance(img_id, str):
            ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=0)
        else:
            ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=0)
        annos = coco.loadAnns(ann_ids)
        if not has_valid_annotation(annos):
            continue

        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        caption = img_info['caption']

        regions = {}

        for anno in annos:
            box = anno['bbox']
            tokens_positive = anno['tokens_positive']
            x1, y1, w, h = box
            inter_w = max(0, min(x1 + w, int(img_info['width'])) - max(x1, 0))
            inter_h = max(0, min(y1 + h, int(img_info['height'])) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if anno['area'] <= 0 or w < 1 or h < 1:
                continue

            if anno.get('iscrowd', False):
                continue
            bbox_xyxy = [
                x1, y1,
                min(x1 + w, int(img_info['width'])),
                min(y1 + h, int(img_info['height']))
            ]

            tokens_positive = sorted(tokens_positive, key=lambda x: x[0])

            phrase = []
            pre_end_index = -10
            for token in tokens_positive:
                start_index = token[0]
                end_index = token[1]
                if pre_end_index + 1 == start_index:
                    if caption[token[0] - 1] == ' ':
                        phrase[
                            -1] = phrase[-1] + ' ' + caption[token[0]:token[1]]
                    else:
                        phrase.append(caption[token[0]:token[1]])
                else:
                    phrase.append(caption[token[0]:token[1]])
                pre_end_index = end_index

            key = ' '.join(phrase)

            if key not in regions:
                regions[key] = {
                    'bbox': bbox_xyxy,
                    'phrase': phrase,
                    'tokens_positive': tokens_positive
                }
            else:
                old_box = regions[key]['bbox']
                if isinstance(old_box[0], list):
                    old_box.append(bbox_xyxy)
                else:
                    old_box = [old_box, bbox_xyxy]

                regions[key]['bbox'] = old_box

        out_dict = {
            'filename': file_name,
            'height': int(img_info['height']),
            'width': int(img_info['width']),
            'grounding': {
                'caption': caption
            }
        }

        region_list = []
        for key, value in regions.items():
            phrase = value['phrase']
            if len(phrase) == 1:
                phrase = phrase[0]
            region_list.append({
                'bbox': value['bbox'],
                'phrase': phrase,
                'tokens_positive': value['tokens_positive']
            })
        out_dict['grounding']['regions'] = region_list
        out_results.append(out_dict)

    if args.out_dir is None:
        out_path = osp.join(args.mdetr_anno_dir, filename[:-5] + '_vg.json')
    else:
        out_path = osp.join(args.out_dir, filename[:-5] + '_vg.json')

    with jsonlines.open(out_path, mode='w') as writer:
        writer.write_all(out_results)
    print(f'save to {out_path}')


def main():
    args = parse_args()
    process_item(args, 'finetune_refcoco_train.json')
    process_item(args, 'finetune_refcoco+_train.json')
    process_item(args, 'finetune_refcocog_train.json')
    process_item(args, 'finetune_grefcoco_train.json')


if __name__ == '__main__':
    main()
