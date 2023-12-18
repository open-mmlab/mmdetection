import argparse
import json
import multiprocessing
import os
import os.path as osp

import emoji
import jsonlines
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
is_debug = False


def is_valid_caption(caption, rules={'↙️', '[CLS]', '[SEP]'}):
    check_anno = caption.strip(
    )[:-1]  # Remove the ending delimiter from the caption.
    for ch in rules:
        if ch in check_anno:
            return False
    return True


def process_one_file(anno_file, result_queue):
    print('processing', anno_file)
    with open(anno_file, 'r') as f:
        metas = json.load(f)

    results = []
    for meta in metas:
        # print('============================')
        file_name = meta['key'][0:5] + '/' + meta['key'] + '.jpg'
        file_name = osp.join('images', file_name)

        h = meta['height']
        w = meta['width']

        caption = meta['caption']
        # Weird captions are filtered out from the beginning.
        if not is_valid_caption(caption):
            if is_debug:
                print('=====caption filtered====', caption)
            continue

        # Captions exceeding 240 tokens are filtered out,
        # where 240 is an empirical value.
        tokenized = tokenizer([caption], return_tensors='pt')
        if tokenized.input_ids.shape[1] >= 240:
            if is_debug:
                print('=====token filtered====', caption)
            continue

        ref_exps = meta['ref_exps']
        ref_captions = [i[0:2] for i in ref_exps]
        ref_token_positives = [i[0:2] for i in ref_exps]
        ref_captions = [caption[int(i[0]):int(i[1])] for i in ref_captions]
        ref_boxes = [i[2:6] for i in ref_exps]

        regions = {}
        for bbox, ref_caption, tokens_positive in zip(ref_boxes, ref_captions,
                                                      ref_token_positives):
            #  If the current reference includes special delimiters,
            #  it will be filtered out.
            if not is_valid_caption(
                    caption, rules={'.', '？', ' ', "\'", "\""}):
                if is_debug:
                    print('=====ref filtered====', caption)
                continue
            # If the current reference contains non-ASCII characters,
            # it will be filtered out.
            if not str.isascii(caption):
                if is_debug:
                    print('=====ref filtered====', caption)
                continue
            # If the current reference includes non-ASCII characters,
            # it will be filtered out.
            if emoji.emoji_count(caption):
                if is_debug:
                    print('=====ref filtered====', caption)
                continue

            box = [
                round(bbox[0] * w, 3),
                round(bbox[1] * h, 3),
                round((bbox[2]) * w, 3),
                round((bbox[3]) * h, 3)
            ]
            x1, y1, x2, y2 = box
            inter_w = max(0, min(x1 + w, int(w)) - max(x1, 0))
            inter_h = max(0, min(y1 + h, int(h)) - max(y1, 0))
            if inter_w * inter_h == 0:
                if is_debug:
                    print('=====wh filtered====', box)
                continue
            if w <= 1 or h <= 1:
                if is_debug:
                    print('=====area filtered====', box)
                continue

            if ref_caption not in regions:
                regions[ref_caption] = {
                    'bbox':
                    box,
                    'phrase':
                    ref_caption,
                    'tokens_positive':
                    [[int(tokens_positive[0]),
                      int(tokens_positive[1])]],
                }
            else:
                old_box = regions[ref_caption]['bbox']
                if isinstance(old_box[0], list):
                    old_box.append(box)
                else:
                    old_box = [old_box, box]
                regions[ref_caption]['bbox'] = old_box

        if len(regions) > 0:
            print('caption: ', caption)
            print('regions', regions)
        else:
            if is_debug:
                print('caption: ', caption)
                print('regions', regions)

        if len(regions) == 0:
            continue

        out_dict = {
            'filename': file_name,
            'height': int(h),
            'width': int(w),
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
        print(out_dict)
        results.append(out_dict)
    result_queue.put(results)


def grit2odvg(args):
    annotations_dir = osp.join(args.data_root, 'annotations')
    annos_files = [
        osp.join(annotations_dir, anno) for anno in os.listdir(annotations_dir)
        if anno.endswith('.json') and not anno.endswith('vg.json')
    ]

    annos_files = annos_files[:2]

    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    pool = multiprocessing.Pool(processes=min(len(annos_files), 16))

    for anno_file in annos_files:
        pool.apply_async(process_one_file, args=(anno_file, result_queue))

    pool.close()
    pool.join()

    out_datas = []
    while not result_queue.empty():
        out_datas.extend(result_queue.get())

    out_path = osp.join(args.data_root, 'grit20m_vg.json')
    with jsonlines.open(out_path, mode='w') as writer:
        writer.write_all(out_datas)
    print('save to ', out_path)
    print('total img: ', len(out_datas))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('grit to odvg format.', add_help=True)
    parser.add_argument('data_root', type=str, help='input dir name')
    args = parser.parse_args()

    grit2odvg(args)
