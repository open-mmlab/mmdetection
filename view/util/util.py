import os
import os.path as osp

# originaldir = './cache/origin/'
# if not os.path.exists(originaldir):
#     os.makedirs(originaldir)
import sys

import cv2
import numpy as np

result_cache_path = './cache/result'
if not os.path.exists(result_cache_path):
    os.makedirs(result_cache_path)

# originpath = osp.join(originaldir,'cache.png')
result_img_cache_path = osp.join(result_cache_path, 'result_cache.png')
result_video_cache_path = osp.join(result_cache_path, 'result_video_cache.mp4')


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path), -1)
    return cv_img


def cv_imwrite(ext, img, save_path):
    cv2.imencode(ext, img)[1].tofile(save_path)


def get_file_name(path):
    return os.path.split(path)[-1]


def img_seq_2_video(dataset_path, output_path):
    # 要被合成的多张图片所在文件夹
    # 路径分隔符最好使用“/”,而不是“\”,“\”本身有转义的意思；或者“\\”也可以。
    # 因为是文件夹，所以最后还要有一个“/”

    list = []
    size_list = []
    for root, dirs, files in os.walk(dataset_path):
        for dir in dirs:
            cur_video = []
            cur_video.append(dir)
            for file in os.listdir(osp.join(dataset_path, dir)):
                cur_video.append(file)  # 获取目录下文件名列表
            list.append(cur_video)

    # img_test = cv_imread(osp.join(img_path,list[0]))
    for video in list:
        video_name = video[0]
        print('写入视频：', video_name)
        # img_test = cv_imread(osp.join(dataset_path,video_name,video[1]))
        img_test = cv2.imread(osp.join(dataset_path, video_name, video[1]))
        img_video_size = (img_test.shape[1],img_test.shape[0])
        size_list.append(img_video_size)

        # VideoWriter是cv2库提供的视频保存方法，将合成的视频保存到该路径中
        # 'MJPG'意思是支持jpg格式图片
        # fps = 5代表视频的帧频为5，如果图片不多，帧频最好设置的小一点
        # (1280,720)是生成的视频像素1280*720，一般要与所使用的图片像素大小一致，否则生成的视频无法播放
        # 定义保存视频目录名称和压缩格式，像素为1280*720
        video_path = osp.join(output_path, video_name + '.mp4')
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, img_video_size)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for i in range(1, len(video)):
            # 读取图片
            img = cv2.imread(osp.join(dataset_path, video_name, video[i]))
            # resize方法是cv2库提供的更改像素大小的方法
            # 将图片转换为1280*720像素大小
            img = cv2.resize(img,img_video_size)
            # 写入视频
            video_writer.write(img)
        print('写入完成')
        # 释放资源
        video_writer.release()
