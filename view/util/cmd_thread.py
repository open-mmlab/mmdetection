import os
import time

from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal


class cmd_thread(QThread):  # 步骤1.创建一个线程实例
    finish_signal = pyqtSignal(str)

    def __init__(self,cmd=None,data_type=None):
        super(cmd_thread, self).__init__()
        self.cmd = cmd
        self.data_type = data_type

    def set_cmd(self, cmd):
        self.cmd = cmd

    def get_cmd(self):
        return self.cmd

    def set_data_type(self, data_type):
        self.data_type = data_type

    def get_data_type(self):
        return self.cmd

    def run(self):
        print("进入后台")
        try:
            os.system(self.cmd)
            print("执行完成")
        except Exception as e:
            print(e)
        self.finish_signal.emit(self.data_type)
        time.sleep(1)
        QtWidgets.QApplication.processEvents()
