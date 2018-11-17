import os
from PIL import Image
#from crnnport import *
from ctpnport import *
import numpy as np
import shutil

path = os.path.abspath(os.curdir)

#对txt文本数据按列排序
def sortText(txt_path):
    f = open(txt_path)
    line = f.readline()
    data_list = []
    while line:
        num = list(map(int,line.split(',')))
        data_list.append(num)
        line = f.readline()
    f.close()
    data_array = np.array(data_list)

    data_array = data_array[data_array[:,1].argsort()]

    np.savetxt(txt_path, data_array, fmt='%i');    

#主函数，连接CTPN和CRNN
if __name__ == '__main__':
    im_name = input("please input file name:")
    #im_name = 'timg.jpg'
    #测试图片的输入路径
    img_path = path + "/data/demo/" + im_name
    im = Image.open(img_path)
    if im is None:
          print ("Error input")
          exit()
    im.show()
    #调用CTPN算法检测文本区域
    ctpnSource(img_path)
  
    im_2 = Image.open(path + "/data/results/" + im_name)
    im_2.show()
    #检测文本区域的坐标保存到txt文件
    base_name = img_path.split('/')[-1]
    txt_path = path + "/data/results/" + "res_{}.txt".format(base_name.split('.')[0])
    sortText(txt_path)
    fp = open(txt_path)
    datas = fp.readlines()
    #index = 0
    #用于字幕识别
    if os.path.exists(path + "/data/output/"):
        shutil.rmtree(path + "/data/output/")
    os.makedirs(path + "/data/output/")
    x_dist = 0
    #index_det = 0
    x_left_det = 0
    y_left_det = 0
    x_right_det = 0
    y_right_det = 0
    for data in datas:
          x_left = int(data.split()[0])
          y_left = int(data.split()[1])
          x_right = int(data.split()[2])
          y_right = int(data.split()[3])
          x_tmp = x_right - x_left
          if x_tmp > x_dist:
              #index_det = index
              x_left_det = x_left
              y_left_det = y_left
              x_right_det = x_right
              y_right_det = y_right
    #裁剪检测到的文本区域，保存为jpg格式，一共有index个区域图片
    region = im.crop((x_left_det, y_left_det, x_right_det, y_right_det))
    region.save(path + "/data/output/" + "0.jpg")
    #index += 1
    region.show()
      
    fp.close()
    
    #调用CRNN算法识别文本
#    for i in range(index):
#        crnnSource("./output/"+str(i)+".jpg")
    
    #用于车牌识别
#    crnnSource("./output/"+str(index_det)+".jpg")
