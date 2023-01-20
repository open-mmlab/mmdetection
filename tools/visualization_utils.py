import numpy as np
from PIL import Image, ImageDraw



def draw_bounding_box_on_image(image, rpn_result):
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    image_pil.show()

    draw = ImageDraw.Draw(image_pil)
    
    for i, rpn in enumerate(rpn_result[0]):
        tl_x, tl_y, br_x, br_y = rpn[0].item(), rpn[1].item(), rpn[2].item(), rpn[3].item()
        draw.rectangle([(tl_x, tl_y), (br_x, br_y)])

    image_pil.show()
    
    return np.asarray(image_pil)
