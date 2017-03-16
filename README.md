# caffe_project
2016年初的人脸识别项目整理，整理收集了十几万网络人脸数据，利用caffe平台，测试了当时流行的caffenet deepID vggnet 等网络，最终利用python实现GUI界面以及简单的注册登录程序。
## 数据
![data_example](https://github.com/VectorSL/caffe_project/blob/master/data_eg.png)
<br>将人脸图片按标签整理，并对人脸区域检测和截取。<br>
数据链接：http://pan.baidu.com/s/1bp4Adu3 密码：ssq4
## 训练
训练部分涉及caffe平台的安装，模型文件的配置，调参等诸多因素，是一个大工程，有时间另行补充。<br>
贴一个验证ROC曲线：<br>
![roc](https://github.com/VectorSL/caffe_project/blob/master/figure_1.png)<br>
效果跟论文中的百分之九十多还是有点差距，原因可能是数据不纯，存在重复数据：即同人不同标签等情况。清洗数据太耗时就不做了。<br>
## 功能实现
本次使用Python语言对模型解析利用，并通过Tkinter模块实现界面化。<br>
功能包括主界面摄像头画面区域，人脸标定框，身份登录按钮Login，重新识别按钮Refresh, 初始化按钮Reset, 退出按钮Quit。<br>
![gui](https://github.com/VectorSL/caffe_project/blob/master/GUI.png)
## 技术指标
* 首次登陆为自动识别及验证，无需人为干预；重新验证需按Refresh按钮重置。<br>
* 摄像头距离头部0.3米至2米内能有效识别，且允许头部有左右45度转动。<br>
* 针对不同的模型识别速度不同，再GPU模式下，识别速度约为0.5s
