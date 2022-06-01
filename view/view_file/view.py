# -*- coding: utf-8 -*-

import os
import shutil
from concurrent.futures import ThreadPoolExecutor

from PyQt5.QtCore import Qt, QPoint
from skimage.io import imread, imsave, imshow
import cv2
from PyQt5 import QtGui, QtCore, QtWidgets
from util.util import result_img_cache_path, cv_imread, cv_imwrite, result_video_cache_path, get_file_name
from util.cmd_thread import cmd_thread

from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QGridLayout, QDialog, QLabel, QWidget, QVBoxLayout, \
    QApplication

from view_file.base_view import Ui_MainWindow


class Ui_view(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(Ui_view, self).__init__(parent)
        self.setupUi(self)

        self.actionOpenImage.triggered.connect(self.open_image)
        self.actionSaveImage.triggered.connect(self.save_result)
        self.actionOpenVideo.triggered.connect(self.open_video)
        # self.actionSaveVideo.triggered.connect(self.save_result_video)
        self.actionExit.triggered.connect(self.close)

        self.pushButtonBaseLine.clicked.connect(self.baseline_detect)
        self.pushButtonLKA.clicked.connect(self.lka_detect)
        self.pushButtonAEMFFM.clicked.connect(self.aem_ffm_detect)

        self.comboBoxDataset.currentIndexChanged.connect(self.dataset_changed)
        self.comboBoxData.currentIndexChanged.connect(self.data_changed)

        # 待检测图片路径
        # self.origin_img_path = ''
        self.ckpt_name = 'checkpoints.pth'
        self.cfg_path = "configs2/{}/{}/{}"
        self.ckpt_path = "checkpoints/{}/{}/{}"
        # 结果视频保存路径
        self.save_video_pos = ''

        self.cwd = os.getcwd()

        self.image_file_types = ('.jpg', '.jpeg', '.png')
        self.image_file_type = "所有文件(*.*);;Jpg Image(*.jpg);;Jpeg Image(*.jpeg);;png Image(*.png)"
        self.video_file_types = ('.mp4', '.mpeg', '.mkv')
        self.video_file_type = "所有文件(*.*);;Mp4 video(*.mp4);;Mpeg video(*.mpeg)"

        self.is_open_image = False
        self.is_open_video = False
        self.cfg_suffixs = {"TinyPerson": "TinyPerson640_newData", "VisDrone": "VisDrone640"}

        self.env_name = 'openmmlab'
        self.env_act_cmd = 'conda activate ' + self.env_name

        self.img_detection_cmd = "python demo/my_image_demo.py " \
                                 "{} " \
                                 "{}" \
                                 " {} " \
                                 "--score-thr 0.7 --save-path {}"

        self.video_detection_cmd = "python demo/video_demo.py " \
                                   "{} " \
                                   "{}" \
                                   " {} " \
                                   "--score-thr 0.7 --out {}"

        self.play_video_cmd = "python view/video_player/videoPlayer.py {}?vName={}"

        self.total_img_detection_cmd = self.env_act_cmd + " && " + self.img_detection_cmd
        self.total_video_detection_cmd = self.env_act_cmd + " && " + self.video_detection_cmd
        self.total_play_video_cmd = self.env_act_cmd + " && " + self.play_video_cmd

        self.datasets = ('请选择', 'TinyPerson', 'VisDrone')
        self.datas_index = ('default', 'image', 'video')
        self.datas = {'default': 'None', 'image': 'None', 'video': 'None'}
        self.cur_dataset = self.datasets[0]
        self.cur_data = self.datas['default']
        self.cur_data_type = 'default'
        self.img_width = 512
        self.img_height = 339

        # 待检测图片路径
        self.origin_img = None
        self.detection_result_img = None

        # 线程池，用来执行终端线程实现异步
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        # 为了返回信号关闭处理中的dialog，这里使用线程实现
        self.cmd_thread = cmd_thread()
        # 提示处理中的小窗口
        self.dialog = QDialog()
        self.set_processing_dialog()
        self.cmd_thread.finish_signal.connect(self.after_detection_event_handler)

        self.init_everything()

    def open_image(self):
        self.is_open_image = False
        filename_choose, _ = QFileDialog.getOpenFileName(self,
                                                         "选择文件",
                                                         self.cwd,  # 起始路径
                                                         self.image_file_type)  # 设置文件扩展名过滤,用双分号间隔

        if filename_choose == "":
            print("取消选择")
            return

        # self.origin_img_path = filename_choose
        self.update_data_path('image', filename_choose)
        file_ext = os.path.splitext(filename_choose)

        if not file_ext[-1] in self.image_file_types:
            self.show_warning("图片类型错误！")
            return

        try:
            self.origin_img = cv_imread(filename_choose)
            # self.origin_img = imread(filename_choose)
        except AttributeError as ae:
            self.show_warning("读取图片文件失败，请检查图片文件！")
            print(ae)
            return

        # if self.is_open_image == False:
        #     self.Init_Widgets()
        self.is_open_image = True
        self.init_origin_view()

    def save_result(self):
        qfd = QFileDialog()
        qfd.setAcceptMode(1)

        filename_choose, filetype = QFileDialog.getSaveFileName(self,
                                                                "选择文件",
                                                                self.cwd,  # 起始路径
                                                                "All Files (*)")  # 设置文件扩展名过滤,用双分号间隔
        if filename_choose == '':
            return

        # result_img_cache_path
        # cv2.imwrite(filename_choose, img=self.detection_result_img)
        ext = os.path.splitext(filename_choose)[-1]
        self.detection_result_img = cv2.cvtColor(self.detection_result_img, cv2.COLOR_BGR2RGB)

        cv_imwrite(ext, img=self.detection_result_img, save_path=filename_choose)
        self.show_info("文件已保存到：" + filename_choose)
        # 清掉缓存图片
        if os.path.exists(result_img_cache_path):
            os.remove(path=result_img_cache_path)
        return

    def baseline_detect(self):
        self.pushButtonBaseLine.setEnabled(False)

        if not self.is_open_image and not self.is_open_video:
            self.show_warning("请先打开一种数据！")
            self.pushButtonBaseLine.setEnabled(True)
            return
        if self.cur_dataset == self.datasets[0]:
            self.show_warning("请先选择一个数据集！")
            self.pushButtonBaseLine.setEnabled(True)
            return

        if self.cur_data == 'default':
            self.show_warning("请选择要检测的数据类型！")
            self.pushButtonBaseLine.setEnabled(True)
            return

        method = 'base'
        cfg = 'faster_rcnn_r50_fpn_1x_{}'.format(self.cfg_suffixs[self.cur_dataset])
        cfg_name = cfg + '.py'
        print('method:', method)
        print("cfg:", self.cfg_path.format(self.cur_dataset, method, cfg_name))
        print("ckpt_path:", self.ckpt_path.format(self.cur_dataset, cfg, self.ckpt_name))

        if self.cur_data == 'image':
            try:
                # self.image_detect(self.origin_img_path,
                self.image_detect(self.datas['image'],
                                  self.cfg_path.format(self.cur_dataset, method, cfg_name),
                                  self.ckpt_path.format(self.cur_dataset, cfg, self.ckpt_name))
            except Exception as e:
                print(e)
                self.show_warning("出现错误，请查看后台输出")
        elif self.cur_data == 'video':
            try:
                # self.image_detect(self.origin_img_path,
                self.video_detect(self.datas['video'],
                                  self.cfg_path.format(self.cur_dataset, method, cfg_name),
                                  self.ckpt_path.format(self.cur_dataset, cfg, self.ckpt_name))
            except Exception as e:
                print(e)
                self.show_warning("出现错误，请查看后台输出")
        self.pushButtonBaseLine.setEnabled(True)

    def lka_detect(self):
        self.pushButtonLKA.setEnabled(False)

        if not self.is_open_image and not self.is_open_video:
            self.show_warning("请先打开一张图片！")
            self.pushButtonLKA.setEnabled(True)
            return
        if self.cur_dataset == self.datasets[0]:
            self.show_warning("请先选择一个数据集！")
            self.pushButtonLKA.setEnabled(True)
            return

        if self.cur_data == 'default':
            self.show_warning("请选择要检测的数据类型！")
            self.pushButtonLKA.setEnabled(True)
            return

        method = 'lka_fpn'
        cfg = 'faster_rcnn_r50_lka_fpn_noaem_noffm_1x_{}'.format(self.cfg_suffixs[self.cur_dataset])
        cfg_name = cfg + '.py'
        print('method:', method)
        print("cfg:", self.cfg_path.format(self.cur_dataset, method, cfg_name))
        print("ckpt_path:", self.ckpt_path.format(self.cur_dataset, cfg, self.ckpt_name))

        if self.cur_data == 'image':
            try:
                # self.image_detect(self.origin_img_path,
                self.image_detect(self.datas['image'],
                                  self.cfg_path.format(self.cur_dataset, method, cfg_name),
                                  self.ckpt_path.format(self.cur_dataset, cfg, self.ckpt_name))
            except Exception as e:
                print(e)
                self.show_warning("出现错误，请查看后台输出")
        elif self.cur_data == 'video':
            try:
                # self.image_detect(self.origin_img_path,
                self.video_detect(self.datas['video'],
                                  self.cfg_path.format(self.cur_dataset, method, cfg_name),
                                  self.ckpt_path.format(self.cur_dataset, cfg, self.ckpt_name))
            except Exception as e:
                print(e)
                self.show_warning("出现错误，请查看后台输出")
        self.pushButtonLKA.setEnabled(True)

    def aem_ffm_detect(self):
        self.pushButtonAEMFFM.setEnabled(False)
        if not self.is_open_image and not self.is_open_video:
            self.show_warning("请先打开一种数据！")
            self.pushButtonAEMFFM.setEnabled(True)
            return
        if self.cur_dataset == self.datasets[0]:
            self.show_warning("请先选择一个数据集！")
            self.pushButtonAEMFFM.setEnabled(True)
            return

        if self.cur_data == 'default':
            self.show_warning("请选择要检测的数据类型！")
            self.pushButtonAEMFFM.setEnabled(True)
            return

        method = 'lka_fpn'
        cfg = 'faster_rcnn_r50_lka_fpn_1x_{}'.format(self.cfg_suffixs[self.cur_dataset])
        cfg_name = cfg + '.py'
        print('method:', method)
        print("cfg:", self.cfg_path.format(self.cur_dataset, method, cfg_name))
        print("ckpt_path:", self.ckpt_path.format(self.cur_dataset, cfg, self.ckpt_name))

        if self.cur_data == 'image':
            try:
                # self.image_detect(self.origin_img_path,
                self.image_detect(self.datas['image'],
                                  self.cfg_path.format(self.cur_dataset, method, cfg_name),
                                  self.ckpt_path.format(self.cur_dataset, cfg, self.ckpt_name))
            except Exception as e:
                print(e)
                self.show_warning("出现错误，请查看后台输出")
        elif self.cur_data == 'video':
            try:
                # self.image_detect(self.origin_img_path,
                self.video_detect(self.datas['video'],
                                  self.cfg_path.format(self.cur_dataset, method, cfg_name),
                                  self.ckpt_path.format(self.cur_dataset, cfg, self.ckpt_name))
            except Exception as e:
                print(e)
                self.show_warning("出现错误，请查看后台输出")
        self.pushButtonAEMFFM.setEnabled(True)

    def image_detect(self, img_path, cfg_path, ckpt_path):
        # 清缓存
        if os.path.exists(result_img_cache_path):
            os.remove(path=result_img_cache_path)
        # 这里调用检测过程
        print("cmd:", self.total_img_detection_cmd.format(img_path, cfg_path, ckpt_path, result_img_cache_path))
        self.show_processing_dialog()
        self.cmd_thread.set_cmd(
            self.total_img_detection_cmd.format(img_path, cfg_path, ckpt_path, result_img_cache_path))
        self.cmd_thread.set_data_type('image')
        self.cmd_thread.start()

        # if self.cmd_thread.finished():
        #     self.dialog.close()

        # os.system(self.total_img_detection_cmd.format(img_path, cfg_path, ckpt_path, result_img_cache_path))

        # result_img = imread(result_img_cache_path)
        # self.detection_result_img = result_img
        # self.set_result_img_view()

    def video_detect(self, video_path, cfg_path, ckpt_path):
        # 清缓存
        QApplication.processEvents()
        if os.path.exists(result_video_cache_path):
            os.remove(path=result_video_cache_path)

        self.show_processing_dialog()
        if self.cur_dataset == 'VisDrone' and self.comboBoxData.currentIndex() == 2:
            # 这里调用检测过程
            print("cmd:",
                  self.total_video_detection_cmd.format(video_path, cfg_path, ckpt_path, self.save_video_pos))

            self.cmd_thread.set_cmd(
                self.total_video_detection_cmd.format(video_path, cfg_path, ckpt_path, self.save_video_pos))
            self.cmd_thread.set_data_type('video')
            self.cmd_thread.start()
            # os.system(self.total_video_detection_cmd.format(video_path, cfg_path, ckpt_path, self.save_video_pos))
        else:
            self.show_warning("暂时只有VisDrone数据集的视频检测功能")
            return

        # result_img = imread(result)
        # self.detection_result_img = result_img
        # self.set_result_img_view()
        # 这里播放刚刚生成的检测结果视频。

    def init_origin_view(self):
        # 缩放进入的图片
        image = QtGui.QPixmap(self.datas['image']).scaled(self.img_width, self.img_height)

        self.labelInput.setPixmap(image)
        self.labelInput.resize(self.img_width, self.img_height)
        self.labelResult.resize(self.img_width, self.img_height)

    def after_detection_event_handler(self, type):
        assert type in self.datas_index
        # 图片
        if type == self.datas_index[1]:
            # 设置图片
            result_img = imread(result_img_cache_path)
            self.detection_result_img = result_img
            # 缩放进入的图片
            image = QtGui.QPixmap(result_img_cache_path).scaled(self.img_width, self.img_height)
            self.labelResult.setPixmap(image)
        # 视频
        elif type == self.datas_index[2]:
            # 播放缓存好的视频
            self.thread_pool.submit(os.system, self.total_play_video_cmd.format(self.save_video_pos,
                                                                                get_file_name(self.save_video_pos)))
            self.show_info("文件已保存到：" + self.save_video_pos)
        self.dialog.close()

    def open_video(self):
        self.is_open_video = False
        filename_choose, _ = QFileDialog.getOpenFileName(self,
                                                         "选择文件",
                                                         self.cwd,  # 起始路径
                                                         self.video_file_type)  # 设置文件扩展名过滤,用双分号间隔

        self.save_video_pos, _ = QFileDialog.getSaveFileName(self,
                                                             "选择保存位置",
                                                             self.cwd,  # 起始路径
                                                             "All Files (*)")  # 设置文件扩展名过滤,用双分号间隔
        if self.save_video_pos == '':
            print("未选择保存位置")
            return

        if filename_choose == "":
            print("取消选择")
            return

        # self.origin_img_path = filename_choose
        self.update_data_path('image', filename_choose)
        file_ext = os.path.splitext(filename_choose)

        if not file_ext[-1] in self.video_file_types:
            self.show_warning("视频类型错误！")
            return

        try:
            # 这里调用播放器播放刚刚打开的视频
            # 异步线程方式
            # self.cmd_thread.set_cmd(cmd=self.total_play_video_cmd.format(filename_choose, file_name))
            # self.cmd_thread.start()
            # 同步方式
            # os.system(self.total_play_video_cmd.format(filename_choose, file_name))
            # 线程池方式
            self.thread_pool.submit(os.system,
                                    self.total_play_video_cmd.format(filename_choose, get_file_name(filename_choose)))
            print('打开视频')
        except AttributeError as ae:
            self.show_warning("读取视频文件失败，请检查视频文件！")
            print(ae)
            return

        self.is_open_video = True
        self.update_data_path('video', filename_choose)
        # self.init_origin_view()

    def init_everything(self):
        self.init_combobox_dataset(self.datasets)
        self.init_combobox_data()

    def init_combobox_dataset(self, datasets):
        for i, dataset in enumerate(datasets):
            self.comboBoxDataset.addItem(dataset)

    def init_combobox_data(self):
        self.comboBoxData.addItem("暂无")
        self.comboBoxData.addItem("图片")
        self.comboBoxData.addItem("视频")

    def dataset_changed(self):
        cur_index = self.comboBoxDataset.currentIndex()
        self.cur_dataset = self.datasets[cur_index]
        print(self.cur_dataset)

    def data_changed(self):
        cur_index = self.comboBoxData.currentIndex()
        self.cur_data = self.datas_index[cur_index]
        print(self.cur_data)

    def update_data_path(self, data_type, path):
        assert data_type in ('image', 'video')
        self.datas[data_type] = path

    def show_info(self, info, buttons=QMessageBox.Close, default_button=QMessageBox.Close):
        QMessageBox.information(self,
                                "提示",
                                info,
                                buttons,
                                default_button)

    def show_warning(self, warning, buttons=QMessageBox.Close, default_button=QMessageBox.Close):
        QMessageBox.warning(self,
                            "警告",
                            warning,
                            buttons,
                            default_button)

    def show_processing_dialog(self):
        # 用exec会阻塞主进程，所用用show
        self.dialog.show()  # 显示对话框

    def set_processing_dialog(self):
        self.dialog.setWindowTitle('处理中')
        self.dialog.setFixedSize(QtCore.QSize(160, 120))
        # 去掉标题栏
        # self.dialog.setWindowFlags(Qt.FramelessWindowHint)
        # 自定义标题栏，只显示标题
        self.dialog.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        self.dialog_label = QLabel('处理中……', self.dialog)

        # 设置居中
        self.dialog_label.setAlignment(Qt.AlignCenter)
        self.dialog_widget = QWidget(self.dialog)
        self.dialog_layout = QVBoxLayout(self.dialog_widget)
        self.dialog_layout.addWidget(self.dialog_label)
        self.dialog.setLayout(self.dialog_layout)

        self.dialog.setWindowModality(Qt.ApplicationModal)  # 当对话框显示时，主窗口的所有控件都不可用

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        # 重写关闭事件，释放线程池资源
        self.thread_pool.shutdown()
        a0.accept()
