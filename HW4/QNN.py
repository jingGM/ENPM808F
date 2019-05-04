import pygame
import numpy as np
import random
import pickle
import csv
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt
from keras.optimizers import SGD,Adam
from keras.models import load_model
import time

class Q_NN():
    def __init__(self,grid_size, epsilon = 0.5,alpha = 0.8, gamma = 0.9):
        #epsilon: possibility to random choose a move from possible moves;  
        #alpha: learning rate;
        #gamma: percentage weight next value;
        self.epsilon=epsilon
        self.alpha=alpha
        self.gamma=gamma
        
        self.Q = {} #Q table

        self.q_last = []
        self.q_last_index=0.0
        self.q_last_state=[]
        
        self.batchsize =32
        self.buffer = []
        
        self.grid_size = grid_size
        outputN = 2*self.grid_size*(self.grid_size+1)
        
        self.model = Sequential()
        self.model.add(Dense(units=64, activation='relu'))
        self.model.add(Dense(units=64, activation='relu'))
        self.model.add(Dense(units=outputN, activation='tanh'))
        self.model.add(Dense(units=outputN, activation='sigmoid'))
        self.sgd = SGD(lr=0.3)
        #self.adam = Adam(lr=0.03)
        self.model.compile(optimizer=self.sgd,loss='mse')
        self.model.predict(np.array([[0]*(outputN+self.grid_size**2)]))
        
    def reset(self):
        self.q_last = []
        self.q_last_index=0
        self.q_last_state=[]
        self.buffer = []
        
    def updateNN(self,reward,Q_state,possiblemoves,won):
        if len(possiblemoves):
            PQstate =np.expand_dims(Q_state, axis=0)
            Q_list =  self.model.predict(PQstate)
            
            #print("PQstate: ",PQstate.shape)
            #print("Q_list: ",Q_list.shape)
            next_q_max = max(Q_list[0][possiblemoves])
        else:
            next_q_max = 0
        Q = self.q_last
        if Q ==[]:
            return
        q_value = Q[self.q_last_index]
        Q[self.q_last_index] = q_value + self.alpha * ((reward + self.gamma*next_q_max) - q_value)
        
        if len(self.buffer)>=self.batchsize or won>0:
            X=[]
            Y=[]
            for i in range(len(self.buffer)):
                X.append(self.buffer[i][0].tolist())
                Y.append(self.buffer[i][1].tolist())
            X=np.array(X)
            #print("X shape: ",X.shape)
            Y=np.array(Y)
            #print("Y shape: ",Y.shape)
            self.cost = self.model.fit(X,Y,epochs=10,verbose=0)
            
            #print(self.cost.history['loss'][0])
            self.buffer = []
        else:
            self.buffer.append([self.q_last_state,Q])
            
    def choosemove(self,Q_state,possiblemoves):
        PQstate =np.expand_dims(Q_state, axis=0)
        Q_list =  self.model.predict(PQstate)
        if (random.random()< self.epsilon):
            move = random.choice(possiblemoves)
        else :
            maxlist = Q_list[0][possiblemoves]
            Qmax = max(maxlist)
            Qmaxlist = [index for index in range(len(maxlist)) if maxlist[index]==Qmax]
            if len(Qmaxlist)==1:
                move = possiblemoves[Qmaxlist[0]]
            else:
                move = possiblemoves[random.choice(Qmaxlist)]
        self.q_last = Q_list[0]
        self.q_last_index = move
        self.q_last_state=Q_state
        return move
    
    def saveQtable(self,file_name):  #save table
        file_name = 'NN'+file_name
        self.model.save(file_name)

    def loadQtable(self,file_name,nfiles): # load table
        file_name = 'NN'+file_name
        self.model = load_model(file_name)