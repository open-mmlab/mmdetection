import jsonlines
from pycocotools.coco import COCO
from tqdm import tqdm
import os

ann_path = '/home/PJLAB/huanghaian/dataset/coco2014/mdetr_annotations/finetune_refcocog_train.json'


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


coco = COCO(ann_path)
ids = list(sorted(coco.imgs.keys()))
out_results = []

i = 0
for img_id in tqdm(ids):
    if i > 1000:
        break
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
    instance_list = []

    for anno in annos:
        box = anno['bbox']

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
        instance_list.append({
            'bbox': bbox_xyxy,
            'exp': caption,
        })

    # 相同图片名的实例合并到一起
    if i != 0 and file_name == out_results[-1]['filename']:
        pre_instance_list = out_results[-1]['referring']['instances']
        for instance in instance_list:
            no_find = True
            for pre_instance in pre_instance_list:
                if instance['bbox'] == pre_instance['bbox'] and instance['exp'] != pre_instance['exp']:
                    if isinstance(pre_instance['exp'], list):
                        pre_instance['exp'].append(instance['exp'])
                    else:
                        pre_instance['exp'] = [pre_instance['exp'], instance['exp']]
                    no_find = False
                    break
            if no_find:
                 pre_instance_list.append(instance)
    else:
        out_results.append({
            'filename': file_name,
            'height': img_info['height'],
            'width': img_info['width'],
            'referring': {
                'instances': instance_list
            }
        })
    i += 1
file_name = os.path.basename(ann_path)
out_path = os.path.join(os.path.dirname(ann_path), os.path.basename(ann_path)[:-5] + '_ref.json')
with jsonlines.open(out_path, mode='w') as writer:
    writer.write_all(out_results)
print(f'save to {out_path}')
