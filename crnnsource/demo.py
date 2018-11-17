import os
from PIL import Image
from crnnport import *
#from ctpnport import *
import numpy as np

path = os.path.abspath(os.curdir)

if __name__ == '__main__':
#	for i in range(index):
#		crnnSource("../ctpnsource/data/output/"+str(i)+".jpg")
    crnnSource("../ctpnsource/data/output/"+"0.jpg")
