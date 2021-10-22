import argparse
import os
import os.path as osp

import cv2
import torch.onnx

#Function to Convert to ONNX
from PIL import Image
from mmcv import Config
from torchvision.transforms import transforms

from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')

    parser.add_argument('config', help='train config file path')
    parser.add_argument('input_path', help='weights .pth file path')
    parser.add_argument('output_path', help='model save path')

    args = parser.parse_args()

    return args

def Convert_ONNX(model,output_path):

    # set the model to inference mode
    model.eval()

    output_dir,file_name = osp.split(output_path)
    if not file_name.endswith(".onnx"):
        raise NameError("Output file must be an onnx file!")

    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    # Let's create a dummy input tensor

    img_size = (640,512)
    dummy_input = torch.zeros(img_size,)

    # Export the model
    torch.onnx.export(model,         # model being run
         dummy_input,       # model input (or a tuple for multiple inputs)
         output_path,       # where to save the model
         export_params=True,  # store the trained parameter weights inside the model file
         # opset_version=10,    # the ONNX version to export the model to
         do_constant_folding=True,  # whether to execute constant folding for optimization
         input_names = ['modelInput'],   # the model's input names
         output_names = ['modelOutput'], # the model's output names
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes
                                'modelOutput' : {0 : 'batch_size'}})
    print(" ")
    print('Model has been converted to ONNX')

if __name__ == '__main__':
    # Let's build our model
    # train(5)
    # print('Finished Training')

    # Test which classes performed well
    # testAccuracy()

    # Let's load the model we just created and test the accuracy per label
    args = parse_args()

    cfg = Config.fromfile(args.config)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    path = args.input_path
    dict_file = torch.load(path)
    # print(dict_file)
    model.load_state_dict(dict_file['state_dict'])

    # Test with batch of images
    # testBatch()
    # Test how the classes performed
    # testClassess()

    # Conversion to ONNX
    Convert_ONNX(model,args.output_path)