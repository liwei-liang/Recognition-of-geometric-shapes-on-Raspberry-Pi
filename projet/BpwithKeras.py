# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 16:51:17 2017
apprentissage avec module Keras
@author: liang,liu,lyu
"""
from BP_V2 import load_samples
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout

PathTrain =  "C:/Users/liang/Desktop/UTT/SY26/projet/trainSet/"
PathTest =  "C:/Users/liang/Desktop/UTT/SY26/projet/testSet/"
model = Sequential()

model.add(Dense(units=40, input_dim=1024))#ajouter une couche caché
#model.add(Dropout(0.2)) #régularisation dropout
model.add(Activation("relu")) 
model.add(Dense(units=6))#ajouter une couche de sortie
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#charger l'ensemble d'entraînement
train_set = load_samples(PathTrain)
x_train = train_set[0:1024,:].T
y_train = train_set[1024:1030,:].T
model.fit(x_train, y_train, epochs=100, batch_size=32)#apprentissage itération de 100 fois, taille de batch est 32
#charger l'ensemble de test
test_set = load_samples(PathTest)
x_test = test_set[0:1024,:].T
y_test = test_set[1024:1030,:].T
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=500)
print(loss_and_metrics)