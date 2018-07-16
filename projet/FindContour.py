# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 12:13:07 2017
chercher le contour， générer l'entrée du modèle
@author: liang,liu,lyu
"""

import cv2
import numpy as np  

'''
filtre
convertir à gris
opération morphologique
binarisation
trouver le contour
definir la zone à couper
redimensionner le zone coupé à 32*32
'''
def draw_contour(ImgPath):
    img =ImgPath
    blured = cv2.blur(img,(5,5))    #flitre passe bas pour effacer les bruit
    #blured = cv2.GaussianBlur(img,(3,3),0) #flitre Gaussian
    gray = cv2.cvtColor(blured,cv2.COLOR_BGR2GRAY) #convertir la figure couleur à gris
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(50, 50))#definir éléments structure
    #Opérations morphologiques：ouverture et fermeture pour supprimer le bruit de fond
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    #opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel) 
    # dilatation, érosion
    #closed = cv2.erode(closed, None, iterations = 4)#érosion de 4 fois
    #closed = cv2.dilate(closed, None, iterations = 4)# dilatation de 4 fois
    seuil=np.max(np.array(gray))/2+np.min(np.array(gray))/2
    ret, binary = cv2.threshold(closed, seuil, 255, cv2.THRESH_BINARY)#binarasation
    image ,contours,hierarchy= cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#chercher le   
    if len(contours) > 0: # si contour existe
        for index in range (len(contours)):
            left, top, largeur, hauteur = cv2.boundingRect(contours[index])#4 paramètres de la positions du contour   
            if largeur>6 and hauteur>6 or index == len(contours):
                image_contour = cv2.drawContours(img,contours,0,(24,81,172),10)# dessiner le contour
                right = 0
                bot = 0
                #à partir les 4 paramètres du contour, définir le zone à couper
                if hauteur > largeur:
                    line_middle = largeur/2+left
                    left = round(line_middle - hauteur/2)
                    right = round(line_middle + hauteur/2)
                    bot = top + hauteur
                elif hauteur <= largeur:
                    col_middle = hauteur/2+top
                    top = round(col_middle - largeur/2)
                    bot = round(col_middle + largeur/2)
                    right = left + largeur
                if(left<0):
                    right = right - left
                    left = 0
                if(top<0):
                    bot = bot - top
                    top = 0
                box = (left,top,right,bot) #definir la zone à couper  
                img_cut = binary[top:bot,left:right]  #couper la figure  
                img_resized = cv2.resize(img_cut,(32, 32))#redimensionner
                return image_contour,img_resized,box
    else:#si contour n'exist pas
        return img,binary[0:32,0:32],(0,0,0,0)

    
    
    
    
    