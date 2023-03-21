import json
import numpy as np

# 读取 JSON 文件
with open('data/wgisd/coco_annotations/new_train_bbox_instances.json', 'r') as f:
    data = json.load(f)

# 定义函数，根据 bbox 生成基于矩形的 segmentation
def bbox_to_segmentation(bbox):
    x, y, w, h = bbox
    # 将 bbox 转换为 COCO 格式的多边形
    segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]
    return segmentation

# 循环处理每个实例
for instance in data['annotations']:
    if not instance['segmentation'] or len(instance['segmentation']) == 0:
        instance['segmentation'] = bbox_to_segmentation(instance['bbox'])

# 将处理后的结果写回 JSON 文件
with open('data/wgisd/coco_annotations/new_train_bbox_instances_p.json', 'w') as f:
    json.dump(data, f)