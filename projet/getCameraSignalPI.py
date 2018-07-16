# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:18:40 2017
fonction principale du projet
@author: liang,liu,lyu
"""
import time
import cv2
from FindContour import draw_contour
import numpy as np
import pandas as pd
from chargerWeight import getWeight

#charger le modèle et juger les formes
def jugerForme(entre_image):
    forme = "";
    w1,w2,b1,b2 = getWeight()
    z1= sigmoid(np.dot(w1,entre_image) + b1)
    z2= sigmoid(np.dot(w2,z1) + b2) # la sortie du modèle
    print(z2) 
    seuil = 0.95 #seuil pour juger de la forme
    result = np.argmax(z2)#trouver l'indice de la valeur maximal de sortie
    if(z2[np.argmax(z2)] > seuil):
        if(result==0): # si l'indice de valeur maximal est 0, un rond est détecté
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
    else: # si la valeur maximal est inférieur au seuil, c'est-à-dire pas de forme détecté
        print("no forme détecté ")
        forme = "no forme detecte"
        return forme
# defini la fonction d'activiation sigmoid
def sigmoid(z):  
        return 1.0/(1.0+np.exp(-z)) 

#écrire le résultat de jugement sur la figure
def writeForme(image,forme,box):        
    font=cv2.FONT_HERSHEY_SIMPLEX
    formeV = forme
    cv2.putText(image,formeV, (100,100), font, 1,(255,255,255),2,cv2.LINE_AA)


if __name__ == '__main__':  
    # démarrage de la caméra
    with picamera.PiCamera() as camera:
        camera.resolution = (640, 480)
        camera.framerate = 24
        time.sleep(1)
        while True:
            #convertir la figure à format de OpenCV numpy array 
            image = np.empty((480 * 640 * 3,), dtype=np.uint8)
            camera.capture(image, format='rgb')
            image = image.reshape((480, 640, 3))
            image_contour,image_cut,box = draw_contour(image) # Appelez la fonction pour trouver le contour
            img_2degree = np.where(image_cut>0, 1,0) #binarisation en valeur 0 et 1
            entree_image = img_2degree.reshape(32*32,1)# redimensionner de figure (32,32) à (1024,1) comme l'entrée du modèle
            forme = jugerForme(entree_image)#juger le résultat obtenu
            writeForme(image_contour,forme,box)# écrire le résultat
            cv2.imshow('frame', image)
            cv2.waitKey(5)
            if cv2.waitKey(1) & 0xFF == ord('q'):#Attend 1 unité de temps, quitte si l'entrée clavier q est détectée
                break


