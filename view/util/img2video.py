import os, sys
import cv2
import numpy as np
import argparse

imgs_path = 'C:\\'

target_size = (1280, 720)
target_fps = 1.0
# 输出文件名
target_video = 'out.mp4'
# 是否保存 resize 的中间图像
saveResizeFlag = False
img_types = ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')

# 不存在则创建目录
def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

# 将图片等比例缩放，不足则填充黑边
def resizeAndPadding(img):
    size = img.shape
    h, w = size[0], size[1]
    target_h, target_w = target_size[1], target_size[0]

    # 确定缩放的尺寸
    scale_h, scale_w= float(h / target_h), float(w / target_w)
    scale = max(scale_h, scale_w)
    new_w, new_h = int(w / scale), int(h / scale)

    # 缩放后其中一条边和目标尺寸一致
    resize_img = cv2.resize(img, (new_w, new_h))

    # 图像上、下、左、右边界分别需要扩充的像素数目
    top = int((target_h - new_h) / 2)
    bottom = target_h - new_h - top
    left = int((target_w - new_w) / 2)
    right = target_w - new_w - left
    # 填充至 target_w * target_h
    pad_img = cv2.copyMakeBorder(resize_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0]) 

    return pad_img
 
 
def imgs2video():
    output_path = imgs_path + 'out\\'
    mkdir(output_path)
    target = output_path + target_video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(target, fourcc, target_fps, target_size)

    images = os.listdir(imgs_path)
    count = 0
    for image in images:
        if not (image.lower().endswith(img_types)):
            continue
        try:
            print(image)
            # cv2.waitKey(100)
            # frame = cv2.imread(imgs_path + image)
            # imread 不能读中文路径，unicode也不行
            frame = cv2.imdecode(np.fromfile(imgs_path + image, dtype=np.uint8), cv2.IMREAD_COLOR) #, cv2.IMREAD_UNCHANGED
            
            pad_frame = resizeAndPadding(frame)
            # print(pad_frame.shape)

            if saveResizeFlag:
                # 保存缩放填充后的图片
                resize_path = imgs_path + 'resize\\'
                mkdir(resize_path)
                resize_name = resize_path + 'resize_' + image
                # cv2.imwrite(resize_name, pad_frame)
                # imwrite 不能读中文路径，unicode也不行
                cv2.imencode(os.path.splitext(image)[-1], pad_frame)[1].tofile(resize_name)
            
            # 写入视频
            vw.write(pad_frame)
            count += 1
        except Exception as exc:
            print(image, exc)
    vw.release()
    print('\r\nConvert Success! Total ' + str(count) + ' images be combined into the video at: ' + target + '\r\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Function: convert images to video")
    parser.add_argument('--input', '-i', required = True)
    parser.add_argument('--output', '-o', default='out.mp4')
    parser.add_argument('--fps', '-f', type=float, default = 1.0)
    parser.add_argument('--resolution', '-r', type=int, nargs = 2, default = [1280, 720])
    parser.add_argument('--save', '-s', action='store_true')
    args = parser.parse_args()
    if args.input:
        if not os.path.isdir(args.input):
            print("input is not a directory")
            sys.exit(0)
        imgs_path = args.input
        if not imgs_path.endswith(('\\', '/')):
            imgs_path += os.path.sep
        print('input path: ' + imgs_path)
    if args.output:
        target_video = args.output
        print('output file: ' + target_video)
    if args.fps:
        if not args.fps > 0:
            print('fps should be greater than zero')
            sys.exit(0)
        target_fps = args.fps
        print('output file fps: ' + str(target_fps))
    if args.resolution:
        if not args.resolution[0] > 0 and args.resolution[1] > 0:
            print('resolution should be greater than zero')
            sys.exit(0)
        target_size = (args.resolution[0], args.resolution[1])
        print('output file resolution: ' + str(target_size))
    if args.save:
        saveResizeFlag = True
    imgs2video()