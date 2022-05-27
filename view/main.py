import multiprocessing
import sys

from PyQt5.QtWidgets import QApplication

from view.view_file.view import Ui_view



# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    ui = Ui_view()
    ui.show()
    sys.exit(app.exec_())




