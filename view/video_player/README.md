# PyQt5VideoPlayer
由于个人的博客系统中在线播放视频功能需要播放不同格式的视频文件，但是目前的H5播放器只支持MP4, WebM, 和 Ogg不能满足需要，如果把所有的视频格式都做统一的转码处理这种方式又太耗时间，所以决定使用了Pyqt5开发了一个简单的多格式的视频播放器，通过使用URL Protocol 方式来通过web页面调用本地应用程序的方式播放在线多格式的视频文件。
* **暂不支持全屏和窗口最大化**
* 安装 LAVFilters-0.74.1-Installer.exe程序做解码器，即可支持avi、MP4、flv、rmvb等视频格式（必须）
* [**pyqt5官方文档**](https://www.riverbankcomputing.com/static/Docs/PyQt5/api/qtwidgets/qaction.html)
* **DevRequird:**

        ➣  Python 3.x
        ➣  pip install pyqt5

* **Run .py File：**

        ➣  Python videoPlayer.py ‘视频路径’?vName=视频名称
      例：
            python videoPlayer.py d:/123.avi?vName=123.avi
            python videoPlayer.py http://127.0.0.1/12345.mp4?vName=12345.mp4  （视频地址是 HTTP 的时候，视频名称最好设置为数字，其他字符会导致视频无法播放，原因暂时未知,所以添加了真实的视频名称用于显示）
* **MakeExeFileRequird:**

        ➣  pip install pywin32
        ➣  pip install pyinstaller
        ➣  打开 cmd, 切换到该项目的根目录下运行： pyinstaller -F -w -i images/favicon.ico videoPlayer.py
      会在目录下生成__pycache__、build、dist三个目录，目录结构如图：
![image](https://github.com/Mr-hongji/PyQt5VideoPlayer/blob/master/images/pyinstaller_ok.png)
![image](https://github.com/Mr-hongji/PyQt5VideoPlayer/blob/master/images/pyinstaller_ok_1.png)

        ➣  运行 dist下的 exe文件，ok。

     
* **Protocol Url使用:**

     ➣  打开目录下的 videoPlayer.reg 文件， 修改EXE的放置路径：
![image](https://github.com/Mr-hongji/PyQt5VideoPlayer/blob/master/images/registerFile.png)

     ➣  双击运行.reg文件， 弹出是否继续提示框，选择 “是”，后提示注册完成。
     ➣  HTML页面中的使用：`<a href="videoPlayer://http://127.0.0.1/3.mp4?vName=3.MP4">videoPlayer 测试</a>`


* **打包关联图标到exe中**
       ➣  在目录先新建 images.qrc 文件，如图：
       
 ![image](https://github.com/Mr-hongji/PyQt5VideoPlayer/blob/master/images/qrc.png)
 
        ➣  打开 cmd， 运行命令：pyrcc5 -o images.py images.qrc,  把.qrc文件转换成.py文件
        ➣  在 videoPlayer.py 文件中导入 images.py (import images)
        ➣  在 videoPlayer.py 文件中引用图片资源文件（:/images/play_btn_icon.png）
        ➣  最后执行 pyinstaler -F -w -i favicon videoPlayer.py  命令生成exe, 图标资源文件就一起被打包到exe里了。

* **说明:**

        ➣ 本地视频文件名中包含中文或空格时，需要把文件名 encode() 一下
        ➣ 在线的视频文件名目前只支持数字，比如：123.mp4

如图：
![image](https://github.com/Mr-hongji/PyQt5VideoPlayer/blob/master/images/videoplayer.jpg)
