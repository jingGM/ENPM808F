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
from DotsNBoxes import DotsNBoxes
from QNN import Q_NN
from Qtable import Q_table

plays = [100,1000,10000]
for pixel in [2,3]:
	Performance = []
    for iterations in plays:
        cplayer1 = Q_table()
        Train = DotsNBoxes(pixel)
        nfiles=Train.train(iterations,cplayer1)
        print("start game")
        Game = DotsNBoxes(pixel)
        Performance.append(Game.randomplay(iterations,100,cplayer1,nfiles))
        
    with open('Performance_table'+str(pixel)+str(iterations)+'.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for i in range(len(Performance)):
                writer.writerow(Performance[i])  
    
    plt.figure()          
    y1 = [Performance[0][1],Performance[1][1],Performance[2][1]]
    y2 = [Performance[0][0],Performance[1][0],Performance[2][0]]
    y3 = [Performance[0][2],Performance[1][2],Performance[2][2]]
    plt.plot(plays,y1,'r',label='win')
    plt.plot(plays,y2,'b',label='loose')
    plt.plot(plays,y3,'g',label='tie')
    plt.legend()
    plt.savefig(str(pixel)+'_table_grids.png')


plays = [100,1000,10000]
for pixel in [2,3]:
    Performance = []
    for iterations in plays:
        cplayer1 = Q_NN(pixel)
        Train = DotsNBoxes(pixel)
        Train.trainNN(iterations,cplayer1)
        print("start game")
        Game = DotsNBoxes(pixel)
        Performance.append(Game.randomplay(iterations,100,cplayer1,1))
        
    with open('Performance_NN'+str(pixel)+str(iterations)+'.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for i in range(len(Performance)):
                writer.writerow(Performance[i])  
    
    plt.figure()          
    y1 = [Performance[0][1],Performance[1][1],Performance[2][1]]
    y2 = [Performance[0][0],Performance[1][0],Performance[2][0]]
    y3 = [Performance[0][2],Performance[1][2],Performance[2][2]]
    plt.plot(plays,y1,'r',label='win')
    plt.plot(plays,y2,'b',label='loose')
    plt.plot(plays,y3,'g',label='tie')
    plt.legend()
    plt.savefig(str(pixel)+'_NN_grids.png')
