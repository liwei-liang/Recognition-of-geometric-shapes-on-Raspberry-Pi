# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:18:40 2017

@author: liang
"""

import cv2
from FindMiddle import findMiddle
from FindMiddleOpenCV import draw_contour
from PIL import Image
from keras.models import load_model
import numpy as np

def jugerForme(entre_image):
    forme = "";
    test=dnn.predict(entree_image.T).reshape(6,1)
    print(test)
    seuil = 0.97
    result = np.argmax(test)
    if(test[np.argmax(test)] > seuil):
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
    
def writeForme(image,forme,box):
    font=cv2.FONT_HERSHEY_SIMPLEX
    formeV = forme
    cv2.putText(image,formeV, (int(box[0]),int(box[1])), font, 1,(255,255,255),2,cv2.LINE_AA)
    #cv2.putText(image,formeV, (100,100), font, 1,(255,255,255),2,cv2.LINE_AA)

    
if __name__ == '__main__':  
#from FindMiddle import findMiddle
        
    cap = cv2.VideoCapture(0)
    i = 0;
    dnn=load_model('C:/Users/liang/Desktop/UTT/SY26/projet/dnn.h5')
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        image_contour,image_cut,box = draw_contour(frame)
        #imageFrame = Image.fromarray(image_closed) 
        #image_cut,box = findMiddle(imageFrame)
        print(box)
        #image_cut.show()
        #img_2degree = image_cut.point(lambda x: 1 if x >= 100 else 0)
        img_2degree = np.where(image_cut>0, 1,0)#binarisation
        #entree_image = np.array(img_2degree).reshape(32*32,1)
        forme = jugerForme(img_2degree)
        writeForme(image_contour,forme,box)
        cv2.imshow('frame', image_contour)
    
        #if(i%60 == 0):
        #    image_cut.show()
        #    print(box)
        #i = i +1
        if cv2.waitKey(1) & 0xFF == ord('q'):#然后等待1个单位时间，如果期间检测到了键盘输入q，则退出，即关闭窗口。
            break
    cap.release()


