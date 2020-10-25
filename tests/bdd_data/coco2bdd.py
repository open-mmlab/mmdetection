import json
from pycocotools.coco import COCO
import argparse


def parse_arguments():
    # parse the arguments
    parser = argparse.ArgumentParser(description='coco to bdd')
    parser.add_argument(
          "--annFile", "-a",
          default="/path/to/coco/label/file",
          help="path to coco label file",
    )
    parser.add_argument(
          "--save_path", "-s"
          default="/save/path",
          help="path to save bdd formatted label file",
    )
    return parser.parse_args()


def transform(annFile):
    # transform to bdd format
    coco = COCO(annFile)
    imgIds = coco.getImgIds()
    imgIds = sorted(imgIds)
    catsIds = coco.getCatIds()
    cats = coco.loadCats(catsIds)
    nms = [cat['name'] for cat in cats]
    catsMap = dict(zip(coco.getCatIds(), nms))
    bdd_label = []
    for imgId in imgIds:
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(annIds)
        det_dict = {}
        det_dict["name"] = img["file_name"]
        det_dict["url"] = img["coco_url"]
        det_dict["attributes"] = {"weather": "undefined",
                                  "scene": "undefined",
                                  "timeofday": "undefined"}
        det_dict["labels"] = []
        for ann in anns:
            label = {"id": ann["id"],
                     "category": catsMap[ann["category_id"]],
                     "manualShape": True,
                     "manualAttributes": True,
                     "box2d": {
                       "x1": ann["bbox"][0],
                       "y1": ann["bbox"][1],
                       "x2": ann["bbox"][0] + ann["bbox"][2] - 1,
                       "y2": ann["bbox"][1] + ann["bbox"][3] - 1,
                     }}
            det_dict["labels"].append(label)
        bdd_label.append(det_dict)
    return bdd_label


def main():
    args = parse_arguments()
    bdd_label = transform(args.annFile)
    with open(args.save_path, 'w') as outfile:
        json.dump(bdd_label, outfile)


if __name__ == '__main__':
    main()
