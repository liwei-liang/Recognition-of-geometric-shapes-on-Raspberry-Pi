# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:44:37 2017

@author: liang
 """
import os
from PIL import Image  

def processImage(filesoure, destsoure, name, imgtype):
    '''
    filesoure是存放待转换图片的目录
    destsoure是存在输出转换后图片的目录
    name是文件名
    imgtype是文件类型
    '''
    imgtype = 'png'

    #打开图片
    im = Image.open(filesoure + name)
    #缩放比例
    im_resized = im.resize((32, 32))  
    im_resized.save(destsoure + name, imgtype)
    im_rotate = im_resized.rotate(30)

    for i in range (1, 11):#  1 to 10
        numero = i
        im_rotate = im_resized.rotate(i*30)
        namer = name.split('.')[0] + "rotate"+ str(numero)+'.png'
        im_rotate.save(destsoure + namer, imgtype)

def main():
    #切换到源目录，遍历源目录下所有图片
    os.chdir(MyPath)
    for i in os.listdir(os.getcwd()):
        #检查后缀
        im = Image.open(i)
        im.show()
        postfix = os.path.splitext(i)[1]
        if postfix == '.PNG' or postfix == '.png':
            processImage(MyPath, OutPath, i, postfix)

if __name__ == '__main__':
    MyPath =  "C:/Users/liang/Desktop/UTT/SY26/projet/samplesOrigine/"
    OutPath = "C:/Users/liang/Desktop/UTT/SY26/projet/samples/"
    main()
    print("done")