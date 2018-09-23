import mmcv
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def coco_eval(result_file, result_types, coco, max_dets=(100, 300, 1000)):
    assert result_file.endswith('.json')
    for res_type in result_types:
        assert res_type in ['proposal', 'bbox', 'segm', 'keypoints']

    if mmcv.is_str(coco):
        coco = COCO(coco)
    assert isinstance(coco, COCO)

    coco_dets = coco.loadRes(result_file)
    img_ids = coco.getImgIds()
    for res_type in result_types:
        iou_type = 'bbox' if res_type == 'proposal' else res_type
        cocoEval = COCOeval(coco, coco_dets, iou_type)
        cocoEval.params.imgIds = img_ids
        if res_type == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.params.maxDets = list(max_dets)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
