# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:44:37 2017
#générer l'ensemble d'apprentissage
@author: liang,liu,lyu
 """
import os
from PIL import Image  
import numpy as np  
from skimage import measure

def processImage(filesoure, destsoure, name, imgtype):
    '''
    filesoure dossier source des figures
    destsourepour dossier où enregistrer les figures après le traitement
    imgtype type de fichier
    '''
    imgtype = 'png'

    im = Image.open(filesoure + name)

    for i in range (0, 72):#  1 to 10
        numero = i
        im_rotate = im.rotate(i*5,False,True)#rotation de la figure
        im_rotate = findMiddle(im_rotate)
        namer = name.split('.')[0] + "rotate"+ str(numero)+'.png'#générer le nom da la nouvelle figure
        im_rotate.save(destsoure + namer, imgtype)

def findMiddle(img):    
    WHITE, BLACK = 255, 0
    img_gray = img.convert("L") #gris
    seuil=np.max(np.array(img_gray))/2
    img_2degree = img_gray.point(lambda x: WHITE if x > seuil else BLACK)

    contours = measure.find_contours(img_2degree, 0)#chercher le contour
    
    right = 0
    left = 0
    bot = 0
    top = 0
    #trouver les quatre coordonnées extrêmement du contour
    for n, contour in enumerate(contours):
        if right == 0 and left == 0 and bot == 0 and top == 0 :
            right = np.max(contour[:,1])
            left = np.min(contour[:,1])+1
            bot = np.max(contour[:,0])
            top = np.min(contour[:,0])+1
        if np.max(contour[:,1])>right:
            right = np.max(contour[:,1])
        if np.min(contour[:,1])<left:
            left = np.min(contour[:,1])+1
        if np.max(contour[:,0])>bot:
            bot = np.max(contour[:,0])
        if np.min(contour[:,0])<top:
            top = np.min(contour[:,0])+1
    #à partir les 4 paramètres du contour, définir le zone à couper
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
     
    box = (left,top,right,bot) #la zone à couper
    img_cut = img.crop(box)  #couper la figure
    img_resized = img_cut.resize((32, 32))# redimensinner
    
    return img_resized

def main():
    #entrer dans le dossier de source qui enregistre les figure originalles
    os.chdir(MyPath)
    for i in os.listdir(os.getcwd()):
        #vérifier le suffixe
        postfix = os.path.splitext(i)[1]
        if postfix == '.PNG' or postfix == '.png':
            processImage(MyPath, OutPath, i, postfix)

if __name__ == '__main__':
    MyPath =  "f:/sy26/projet/samplesOrigine/"
    OutPath = "f:/sy26/projet/total_sample/"
    main()