import numpy as np
from PIL import Image, ImageDraw,ImageFont



def draw_bounding_box_on_image(image, rpn_result):
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')

    draw = ImageDraw.Draw(image_pil)
    
    for i, rpn in enumerate(rpn_result):
        tl_x, tl_y, br_x, br_y = rpn[0].item(), rpn[1].item(), rpn[2].item(), rpn[3].item()
        draw.rectangle([(tl_x, tl_y), (br_x, br_y)])
    
    return np.asarray(image_pil)

def draw_bounding_box_50_on_image(image, rpn_result):
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')

    draw = ImageDraw.Draw(image_pil)

    for i, rpn in enumerate(rpn_result):
        tl_x, tl_y, br_x, br_y = rpn[0].item(), rpn[1].item(), rpn[2].item(), rpn[3].item()
        draw.rectangle([(tl_x, tl_y), (br_x, br_y)])
        if i == 50: break
    
    return np.asarray(image_pil)

def draw_gt_bounding_box_on_image(image, gt_bboxes, gt_labels, CLASSES, PALETTE):
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')

    draw = ImageDraw.Draw(image_pil)

    for i, (gt_bbox, gt_label) in enumerate(zip(gt_bboxes, gt_labels)):
        tl_x, tl_y, br_x, br_y = gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]
        color = PALETTE[gt_label]
        text = CLASSES[gt_label]
        draw.rectangle([(tl_x, tl_y), (br_x, br_y)], outline=color, width=3)
        draw.text((tl_x+5, tl_y+5), text)
    
    return np.asarray(image_pil)