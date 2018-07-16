# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 21:53:41 2017
BP apprentissage pour la reconnaissance de forme géométrique 
@author: liang
"""
import os
import numpy as np  
from matplotlib import pyplot  
from PIL import Image
class NeuralNet(object):  
    # 初始化神经网络，sizes是神经网络的层数和每层神经元个数  
    def __init__(self, sizes):  
        self.sizes_ = sizes  
        self.num_layers_ = len(sizes)  # nombre de couche, on a choisi 3  
        self.w1_ = np.random.randn(sizes[1], sizes[0])# w_、b_初始化为正态分布随机数  
        self.w2_ = np.random.randn(sizes[2], sizes[1])# w_、b_初始化为正态分布随机数  
        self.b1_ = np.random.randn(sizes[1], 1) 
        self.b2_ = np.random.randn(sizes[2], 1) 

        # Sigmoid函数，S型曲线，  
# Sigmoid函数，S型曲线，  
    def sigmoid(self, z):  
        return 1.0/(1.0+np.exp(-z))  
    # Sigmoid函数的导函数  
    def sigmoid_prime(self, z):  
        return self.sigmoid(z)*(1-self.sigmoid(z))    
    def ReLU(self,z): #fonction d'activation
        zr = np.zeros(z.shape)
        n = z.shape[0] 
        for i in range (0,n):
            z1 = max(0,z[i])
            zr[i]= z1
        return zr
    def ReLU_prime(self,z): #derive de RELU
        zr = np.zeros(z.shape)
        n = z.shape[0]
        for i in range (0,n):
            if(z[i]<=0):
                zr[i] = 0
            else:
                zr[i] = 1
        return zr
    def feedforward(self, x): # x entrée des neurones  
            x = self.sigmoid(np.dot(self.w1_, x)+self.b1_)  
            #print(x)
            x = self.sigmoid(np.dot(self.w2_, x)+self.b2_)  
            return x  #sorties du reseau
    
    #la dérive de la fonction de perte  fonction de perte:L = 1/2 * (Y - Z)**2
    def cost_derivative(self, output_activations, y):  
        output_activations = np.array(output_activations)
        y = np.array(y)

        return output_activations-y  
    
    #x l'entré du reseau y sortie, alpha est le ratio d'apprentissage
    def backprop(self, x, y, alpha):  
        x=x.reshape(1024,1)
        y=y.reshape(5,1)
        d_b1 = np.zeros(self.b1_.shape)
        d_w1 = np.zeros(self.w1_.shape) 
        d_b2 = np.zeros(self.b2_.shape) 
        d_w2 = np.zeros(self.w2_.shape)
        
        z1 = np.dot(self.w1_,x).reshape(self.w1_.shape[0],1) + self.b1_
        s1 = self.sigmoid(z1) #sortie du couche caché apres ReLU
        z2 = np.dot(self.w2_,s1) + self.b2_
        s2 = self.sigmoid(z2) #sortie du réseau 
   
        delta = self.cost_derivative(s2, y) * self.sigmoid_prime(z2) 
        d_b2 = delta
        d_w2 = np.dot(delta, s1.transpose())  
        #ensuite calcule w et b de première couche 
        delta2 = np.dot(self.w2_.transpose(), delta) * self.sigmoid_prime(z1)
        d_b1 = delta2
        d_w1 = np.dot(delta2,x.transpose())
        self.w1_ = self.w1_ - alpha * d_w1
        self.w2_ = self.w2_ - alpha * d_w2
        self.b1_ = self.b1_ - alpha * d_b1
        self.b2_ = self.b2_ - alpha * d_b2
        return (self.w1_, self.w2_, self.b1_, self.b2_)
    
    # training_data是训练数据(x, y); epochs是训练次数; mini_batch_size是每次训练样本数; alpha是learning rate  test_data： Training set
    def SGD(self, training_set, iteration , alpha, test_set=None):  
        if test_set.any():  
            n_test = test_set.shape[1]  
   
        n = training_set.shape[1]  
        for j in range(iteration):  
            training_data = [training_set[:,k] for k in range(0, n, 1)]  
            for data in training_data:  
                self.backprop(data[0:1024], data[1024:1029], alpha)  
            if test_set.any():  
                print("iteration {0}: {1} / {2}".format(j, self.evaluate(test_set), n_test))  
            else:  
                print("iteration {0} complete".format(j))  
                
    #chercher index de la valeur maximale dans la sortie du réseaux et le compare avec la sortie qu'on veut             
    def evaluate(self, test_set):  
        correct = 0
        n_test = test_set.shape[1] 
        for k in range(0, n_test, 1):
            x = test_set[:,k].reshape(1029,1)
            result_feed_forward = np.argmax(self.feedforward(x[0:1024]))
            #print("result_feed_forward:{0}".format(result_feed_forward))
            label = np.argmax((x)[1024:1029])
            #print("label:{0}".format(label))
            #print(correct)
            if result_feed_forward == label:
                correct = correct + 1
        return correct
    
    def predict(self, data):  
        value = self.feedforward(data)  
        return value.tolist().index(max(value)) 
    
    def save(self):  
        np.savetxt("w1.txt",self.w1_)
        np.savetxt("w2.txt",self.w2_)
        np.savetxt("b1.txt",self.b1_)
        np.savetxt("b2.txt",self.b2_)
        # 把_w和_b保存到文件(pickle)  
    def load(self):  
        pass 

def output_geometrique(name): #définir la sortie but du réseau (label)
    y = np.zeros((5,1), dtype=np.int)
    if 'rond' in name:
        y[0] = 1
    if 'carre' in name:
        y[1] = 1
    if 'dix' in name:
        y[2] = 1
    if 'triangle' in name:
        y[3] = 1
    if 'diamond' in name:
        y[4] = 1
    #if 'hexagon' in name:
    #    y[5] = 1
    return y
        
    
def load_samples(path):  
    result = np.zeros((1029,1),dtype=np.int)  # 创建一个空的一维数组
    os.chdir(path)
    for i in os.listdir(os.getcwd()):
        dataTrain_et_label = np.array([])
        geometriqueType = os.path.splitext(i)[0]#prendre le nom 
        y = output_geometrique(geometriqueType)
        image=Image.open(i)
        r, g, b = image.split()
        data_train = np.array(r).reshape(32*32,1) #reshape les entrees en vecteur
        data_train = np.around((data_train+110)/255) #Binarisation de pixels
        dataTrain_et_label = np.vstack((data_train,y))#ajouter lable
        result = np.hstack((result,dataTrain_et_label)) #Empilez les entrées vers array resule horizontalement 
    result = np.delete(result, 0, 1)
    return result

    #shufle tainset
def set_shuffle(set):  
    set = set.T
    np.random.shuffle(set)
    set = set.T
    return set

if __name__ == '__main__':  
    INPUT = 32 * 32 #pixel 32*32
    neurones_caches = 40
    OUTPUT = 5
    PathTrain =  "C:/Users/liang/Desktop/UTT/SY26/projet/trainSet/"
    PathTest =  "C:/Users/liang/Desktop/UTT/SY26/projet/testSet/"

    net = NeuralNet([INPUT, neurones_caches, OUTPUT])  
   
    train_set = load_samples(PathTrain)
    test_set = load_samples(PathTest)

    #shufle tainset
    train_set =set_shuffle(train_set)
    net.SGD(train_set, 1000, 0.1, test_set = test_set)  
    
    
    
    
    
    
    