# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:18:40 2017

@author: liang
"""

import cv2
from chargerWeight import getWeight
from FindContour import draw_contour
from keras.models import load_model
import numpy as np

def jugerForme(entre_image):
    forme = "";
    w1,w2,b1,b2 = getWeight()
    z1= sigmoid(np.dot(w1,entre_image) + b1)
    z2= sigmoid(np.dot(w2,z1) + b2)
    print(z2)
    seuil = 0.95
    result = np.argmax(z2)
    if(z2[np.argmax(z2)] > seuil):
        if(result==0):
            print("rond détecteé ")
            forme = "rond"
            return forme
        if(result==1):
            print("carré détecteé  ")
            forme = "carre"
            return forme
        if(result==2):
            print("dix détecteé ")
            forme = "dix"
            return forme
        if(result==3):
            print("triangle détecteé ")
            forme = "triangle"
            return forme
        if(result==4):
            print("losange détecteé ")
            forme = "losange"
            return forme
        if(result==5):
            print("octogone détecteé ")
            forme = "octogone"
            return forme
    else:
        print("no forme détecté ")
        forme = "no forme detecte"
        return forme
def sigmoid(z):  
        return 1.0/(1.0+np.exp(-z)) 
    
def writeForme(image,forme,box):
    font=cv2.FONT_HERSHEY_SIMPLEX
    formeV = forme
    cv2.putText(image,formeV, (int(box[0]),int(box[1]+30)), font, 1,(255,255,255),2,cv2.LINE_AA)
    #cv2.putText(image,formeV, (100,100), font, 1,(255,255,255),2,cv2.LINE_AA)

    
if __name__ == '__main__':
#from FindMiddle import findMiddle
    i=0
    cap = cv2.VideoCapture(0)
    dnn=load_model('C:/Users/liang/Desktop/UTT/SY26/projet/dnn.h5')  
    while(cap.isOpened()):
        ret, frame = cap.read()
        #image_contour,image_cut,box,closed,opened,gray,binary = draw_contour(frame)
        image_contour,image_cut,box= draw_contour(frame)
        #imageFrame = Image.fromarray(image_closed) 
        #image_cut,box = findMiddle(imageFrame)
        print(box)
        img_2degree = np.where(image_cut>0, 1,0)#binarisation de 0 et 1
        entree_image = img_2degree.reshape(32*32,1)
        forme = jugerForme(entree_image)
        writeForme(image_contour,forme,box)
        cv2.imshow('frame', image_contour)
    
# =============================================================================
#         if(i%60 == 0):
#             cv2.imshow('closed', closed)
#             cv2.imshow('frame2', opened)
#             cv2.imshow('gray', gray)
#             cv2.imshow('binary', binary)
#             print(box)
#         i = i +1
# =============================================================================
        if cv2.waitKey(1) & 0xFF == ord('q'):#然后等待1个单位时间，如果期间检测到了键盘输入q，则退出，即关闭窗口。
            break
    cap.release()


