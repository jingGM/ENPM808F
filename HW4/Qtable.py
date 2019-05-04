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

class Q_table:
    def __init__(self, epsilon = 0.5,alpha = 0.8, gamma = 0.9):
        #epsilon: possibility to random choose a move from possible moves;  
        #alpha: learning rate;
        #gamma: percentage weight next value;
        self.epsilon=epsilon
        self.alpha=alpha
        self.gamma=gamma
        
        self.Q = {} #Q table
        
        self.q_last=0.0
        self.last_Q_index=None
    
    def reset(self):
        self.q_last = 0.0
        self.last_Q_index = None
    
    def getQ(self,state,action):
        move = np.array([action])
        Q_index = tuple(np.concatenate((state,move)))
        if self.Q.get(Q_index) is None:
            self.Q[Q_index] = 0
        return self.Q.get(Q_index)
    
    def choosemove(self,state,possible_moves):
        #state: (edgestatus,gridstatus)
        if (random.random()< self.epsilon):
            move = random.choice(possible_moves)
        else :
            Q_list = [self.getQ(state,move) for move in possible_moves]
            Qmax = max(Q_list)
            Qmaxlist = [index for index in range(len(Q_list)) if Q_list[index]==Qmax]
            if len(Qmaxlist)==1:
                move = possible_moves[Qmaxlist[0]]
            else:
                move = possible_moves[random.choice(Qmaxlist)]
        self.last_Q_index = tuple(np.concatenate((state,np.array([move]))))
        self.q_last = self.getQ(state,move)
        return move
    
    def updateQ (self,reward,state,possible_moves):
        if len(possible_moves):
            Q_list =  [self.getQ(state,move) for move in possible_moves]
            next_q_max = max(Q_list)
        else:
            next_q_max = 0
        self.Q[self.last_Q_index] = self.q_last + self.alpha * ((reward + self.gamma*next_q_max) - self.q_last)
        
        #print("here is update: ")
        #print(self.last_Q_index,self.Q[self.last_Q_index])
    def saveQtable(self,file_name):  #save table
        print(len(self.Q))
        nfiles=0
        if len(self.Q)<2000:
            with open(file_name, 'wb') as handle:
                pickle.dump(self.Q, handle)#, protocol=pickle.HIGHEST_PROTOCOL)
                #print(len(self.Q))
                nfiles=1
        else :
            k = len(self.Q)//2000
            remain = len(self.Q)%2000
            for i in range(k):
                with open(file_name+str(i), 'wb') as handle:
                    pickle.dump(list(self.Q.items())[i*2000:2000*(i+1)], handle)#, protocol=pickle.HIGHEST_PROTOCOL)
                #print(len(self.Q))
                    nfiles+=1
            if remain:
                with open(file_name+str(nfiles), 'wb') as handle:
                    pickle.dump(list(self.Q.items())[nfiles*2000:], handle)#, protocol=pickle.HIGHEST_PROTOCOL)
                    nfiles+=1
        return nfiles
        #w = csv.writer(open(file_name, "w"))
        #for key, val in self.Q.items():
        #    w.writerow([key, val])

    def loadQtable(self,file_name,nfiles): # load table
        if nfiles==1:
            with open(file_name, 'rb') as handle:
                self.Q = pickle.load(handle)
        else :
            for i in range(nfiles):
                with open(file_name+str(i), 'rb') as handle:
                    K=pickle.load(handle)
                    for i in range(len(K)):
                        self.Q.update({K[i][0]:K[i][1]})
        print(len(self.Q))