# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 15:52:01 2017
recharger w et b du modèle
@author: liang,liu,lyu
"""

import pandas as pd
import numpy as np

#charger les weights du modèle
def getWeight():
    w1=pd.read_csv('C:/Users/liang/Desktop/UTT/SY26/projet/w1.csv',encoding = 'gbk') 
    w2=pd.read_csv('C:/Users/liang/Desktop/UTT/SY26/projet/w2.csv',encoding = 'gbk')
    b1=pd.read_csv('C:/Users/liang/Desktop/UTT/SY26/projet/b1.csv',encoding = 'gbk') 
    b2=pd.read_csv('C:/Users/liang/Desktop/UTT/SY26/projet/b2.csv',encoding = 'gbk') 
    w1 = w1.values # convertir la données dataframe vers numpy array
    w1 = np.delete(w1, 0 , axis =1) # supprimer la première colonne qui est indice de la ligne
    w2 = w2.values
    w2 = np.delete(w2, 0 , axis =1) # supprimer la première colonne qui est indice de la ligne
    b1 = b1.values
    b1 = np.delete(b1, 0 , axis =1) # supprimer la première colonne qui est indice de la ligne
    b2 = b2.values
    b2 = np.delete(b2, 0 , axis =1) # supprimer la première colonne qui est indice de la ligne
    return w1,w2,b1,b2
    

if __name__ == '__main__':  
    w1,w2,b1,b2 = getWeight()
    print(w1.shape)
    print(w2)
    print(b1)
    print(b2)