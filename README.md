# OCR_caption_recognition
project: OCR   author: cwx
========================================
功能
========================================
视频字幕识别

========================================
环境
========================================
tensorflow1.6
pytorch0.4
CUDA9.0
CUDNN7.0
python3.5 或者 3.6
安装lmdb: pip install lmdb

========================================
首次使用需要编译
========================================
(a) cd ctpnsource/lib/utils and execute: python setup.py build
(b) copy the .so file from the "build/XXX" directory to the ctpnsource/lib/utils

========================================
运行
========================================
sh setup.sh

然后输入文件名，如：
ren_1.jpg

========================================
结果
========================================
demo1:
![image text](https://github.com/cwxcode/OCR_caption_recognition/master/OCR_caption_recognition/image/ren_1.jpg)

demo2:
![image text](https://github.com/cwxcode/OCR_caption_recognition/master/OCR_caption_recognition/image/ren_2.jpg)
