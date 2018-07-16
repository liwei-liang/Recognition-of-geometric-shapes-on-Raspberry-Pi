# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 23:30:37 2017

@author: lenovo
"""

from skimage import io
import os
import numpy as np  
from PIL import Image
import matplotlib.pyplot as plt
from skimage import measure,data,color

'''
确定图片原始长宽
二值化
可以利用描边的那个方法contour
确定图形的身高和宽度
找到四个点在矩阵中的位置
对图像进行切割
然后再缩小成32*32
'''
def findMiddle(ImgPath):
    #img = Image.open(ImgPath)
    img =ImgPath
    #不同颜色的r g b合适的值都不一样吗所以要先变灰度
    
    WHITE, BLACK = 255, 0
    #如果边界的颜色比较浅会直接变成边界的颜色
    #img_gray = img.convert("L") #gray
    seuil=np.max(np.array(img))/2+np.min(np.array(img))/2
    #seuil= np.max(np.array(img_gray))/2 - 7
    img_2degree = img.point(lambda x: WHITE if x > seuil else BLACK)
    
    contours = measure.find_contours(ImgPath, 0)
    
    right = 0
    left = 0
    bot = 0
    top = 0
    for n, contour in enumerate(contours):
        if right == 0 and left == 0 and bot == 0 and top == 0 :
            right = np.max(contour[:,1])
            left = np.min(contour[:,1])
            bot = np.max(contour[:,0])
            top = np.min(contour[:,0])
        if np.max(contour[:,1])>right:
            right = np.max(contour[:,1])
        if np.min(contour[:,1])<left:
            left = np.min(contour[:,1])
        if np.max(contour[:,0])>bot:
            bot = np.max(contour[:,0])
        if np.min(contour[:,0])<top:
            top = np.min(contour[:,0])
    
    hauteur = bot-top
    largeur = right-left 

    if hauteur > largeur:
        line_middle = largeur/2+left
        left = line_middle - hauteur/2
        right = line_middle + hauteur/2
    elif hauteur < largeur:
        col_middle = hauteur/2+top
        top = col_middle - largeur/2
        bot = col_middle + largeur/2
    box = (left,top,right,bot) #设定裁剪区域  
        
    img_cut = img_2degree.crop(box)  #裁剪图片，并获取句柄region  
    img_resized = img_cut.resize((32, 32))     

    return img_resized,box

    
    
    
    
    
    