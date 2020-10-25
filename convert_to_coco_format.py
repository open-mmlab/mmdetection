# -*- coding: utf-8 -*-

import os
import argparse
import json
import sys
from PIL import Image
from show_process import ShowProcess

"""the coco dataset's json is a dict, and  its keys is categories, images, annotations;
   the categories is a list, and each item is a dict which keys are supercategory,id,name;
   the images is a list, and each item is a dict which keys  are file_name,height,width,id,depth;
   the annotations is a list, and each item is a dict which keys are category_id,segmentation,area,id,iscrowd"""

filenames = []
category_names = ["bike","bus", "car", "motor", "person", "rider", "traffic light", "traffic sign", "train", "truck"]

def parse_args():
    parser = argparse.ArgumentParser(description='convert the file to coco format')
    parser.add_argument(
        '--src',
        dest='src_path',
        help='/path/to/source',
        default=None, type=str
    )
    parser.add_argument(
        '--dst',
        dest='dst_path',
        help='/path/to/save.json',
        default=None,
        type=str
    )
    parser.add_argument(
        '--dir',
        dest='im_dir',
        help='/path/to/image/',
        default=None,
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def visual_bbox(im_path, bbox):
    im = cv2.imread(im_path)
    bbox = [int(i) for i in bbox]
    point1 = (bbox[0],bbox[1])
    point2 = (bbox[2],bbox[3])
    cv2.rectangle(im,point1,point2,(255,0,0))
    return im

def make_coco_categories():
    categoriesList = []
    for i in range(len(category_names)):
        eachcategoryDict = {}
        eachcategoryDict['supercategory'] = 'none'
        eachcategoryDict['id'] = i + 1
        eachcategoryDict['name'] = category_names[i]
        categoriesList.append(eachcategoryDict)
    return categoriesList

def make_coco_images(src_json_file):
    imagesList = []
    global filenames
    for i in range(len(src_json_file)):
        anno = src_json_file[i]
        filename = anno['name']
        filenames.append(filename)
    filenames = list(set(filenames))
    print "it is make_coco_images......."
    process = ShowProcess(len(filenames))
    for index in range(len(filenames)):
        process.show_process()
        eachImageDict = dict()
        filename = filenames[index]
        filepath = os.path.join(args.im_dir,filename)
        assert os.path.isfile(filepath),"{} not is a file".format(filepath)
        #im = cv2.imread(filepath)
        #height,width,depth = im.shape
        im = Image.open(filepath)
        width,height = im.size
        eachImageDict['height'] = height
        eachImageDict['width'] = width
        #eachImageDict['depth'] = depth
        eachImageDict['id'] = index
        eachImageDict['file_name'] = filename
        imagesList.append(eachImageDict)
    return imagesList

def make_coco_annotations(src_json_file):
    global filenames
    annotationsList = []
    print "it is make_coco_annotations....."
    process = ShowProcess(len(src_json_file))
    for i in range(len(src_json_file)):
        process.show_process()
        eachAnnotationDict = dict()
        anno = src_json_file[i]
        category = anno['category']
        bbox = anno['bbox']
        filename = anno['name']
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        xmin = bbox[0]
        ymin = bbox[1]
        box = [xmin, ymin, bbox_width, bbox_height]
        eachAnnotationDict['image_id'] = filenames.index(filename)
        eachAnnotationDict['bbox'] = box
        eachAnnotationDict['category_id'] = category_names.index(category) + 1
        eachAnnotationDict['segmentation'] = [[0,0]]
        eachAnnotationDict['area'] = 1
        eachAnnotationDict['id'] = i
        eachAnnotationDict['iscrowd'] = 0
        annotationsList.append(eachAnnotationDict)
    return annotationsList

def convert_coco(args):
    src = json.load(open(args.src_path))
    assert(type(src) == list),"unsupported type"
    allInfo = dict()
    allInfo['categories'] = make_coco_categories()
    allInfo['images'] = make_coco_images(src)
    allInfo['annotations'] = make_coco_annotations(src)
    print allInfo
    save_path = args.dst_path
    with open(save_path,'w') as writer:
        json.dump(allInfo,writer)

def main(args):
    convert_coco(args)

if  __name__ == "__main__":
    args = parse_args()
    main(args)
