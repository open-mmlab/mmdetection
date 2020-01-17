# encoding:utf/8
import sys
from mmdet.apis import inference_detector, init_detector
import json
import os
import numpy as np
import argparse
from tqdm import tqdm


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

        # generate result


def result_from_dir():
    index = {1: 1, 2: 9, 3: 5, 4: 5, 5: 4, 6: 2, 7: 8, 8: 6, 9: 10, 10: 7}
    # build the model from a config file and a checkpoint file
    model = init_detector(config2make_json, model2make_json, device='cuda:0')
    pics = os.listdir(pic_path)
    meta = {}
    images = []
    annotations = []
    num = 0
    for im in tqdm(pics):
        num += 1
        img = os.path.join(pic_path, im)
        result_ = inference_detector(model, img)
        images_anno = {}
        images_anno['file_name'] = im
        images_anno['id'] = str(num)
        images.append(images_anno)
        for i, boxes in enumerate(result_, 1):
            if len(boxes):
                defect_label = index[i]
                for box in boxes:
                    anno = {}
                    anno['image_id'] = str(num)
                    anno['category_id'] = defect_label
                    anno['bbox'] = [round(float(i), 2) for i in box[0:4]]
                    anno['bbox'][2] = anno['bbox'][2] - anno['bbox'][0]
                    anno['bbox'][3] = anno['bbox'][3] - anno['bbox'][1]
                    anno['score'] = float(box[4])
                    annotations.append(anno)
    meta['images'] = images
    meta['annotations'] = annotations
    with open(json_out_path, 'w') as fp:
        json.dump(meta, fp, cls=MyEncoder, indent=4, separators=(',', ': '))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate result")
    parser.add_argument("-m", "--model", help="Model path", type=str, )
    parser.add_argument("-c", "--config", help="Config path", type=str, )
    parser.add_argument("-im", "--im_dir", help="Image path", type=str, )
    parser.add_argument('-o', "--out", help="Save path", type=str, )
    args = parser.parse_args()
    model2make_json = args.model
    config2make_json = args.config
    json_out_path = args.out
    pic_path = args.im_dir
    result_from_dir()