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


class DotsNBoxes:
    def __init__(self,grid_size=2):
        self.grid_size = grid_size  # default

        self.start_walls = int(0.5 * self.grid_size ** 2)
        self.accept_clicks = True
        
        self.GRIDPIXEL = 100
        self.BOUNDARYPIXEL = 40
        self.DOTWIDTH = 20
    
        # variables for the boxes for each player (x would be computer)
        self.h_boxes = 0
        self.c_boxes = 0
        
        self.training = True
        
        self.turn = "H"
        
        
        # 0 empty 1 is H 2 is C
        self.grid_status = np.zeros(self.grid_size*self.grid_size, np.int)
        self.edge_status = np.zeros(2*self.grid_size*(self.grid_size+1), np.dtype(bool))        
        # o---o   o    [  [ True, False ],
        # |               [ True, False, False], 
        # o   o---o       [ False, True ],
        #     |           [ False, True, False],
        # o---o---o       [ True, True]           ]

        
        
    
    def checkfouredges(self, gridindex):
        row = gridindex // self.grid_size
        col = gridindex % self.grid_size
        upindex = row * (2*self.grid_size+1) + col
        underindex = upindex + (2*self.grid_size+1)
        leftindex = upindex + self.grid_size
        rightindex = leftindex+1
        if (self.edge_status[upindex] and self.edge_status[underindex] and 
            self.edge_status[leftindex] and self.edge_status[rightindex]):
            return True
        else: return False
        
    def checkclosedbox(self, edgeindex):
        number = 0
        for i in range(self.grid_status.shape[0]):
            if self.grid_status[i] == 0:
                if self.checkfouredges(i):
                    number += 1
                    if self.turn == "H":
                        self.grid_status[i] = -1
                        self.h_boxes += 1
                        if not self.training:
                            row = i //self.grid_size
                            col = i % self.grid_size
                            self.screen.blit(self.H, (col * self.GRIDPIXEL+self.BOUNDARYPIXEL,
                                                      row * self.GRIDPIXEL+self.BOUNDARYPIXEL))
                    elif self.turn == "C":
                        self.grid_status[i] = 1
                        self.c_boxes += 1
                        if not self.training:
                            row = i //self.grid_size
                            col = i % self.grid_size
                            self.screen.blit(self.C, (col * self.GRIDPIXEL+self.BOUNDARYPIXEL,
                                                      row * self.GRIDPIXEL+self.BOUNDARYPIXEL)) 
        return number
    
    def won(self):
        """
        Check whether the game was finished
        If so change the caption to display the winner
        :return: won or not
        """
        won = 0
        if self.h_boxes + self.c_boxes == self.grid_size ** 2:
            if self.h_boxes < self.c_boxes:
                won_caption = "Player C won!   Congrats"
                won = 2
            elif self.c_boxes < self.h_boxes:
                won_caption = "Player H won!   Congrats"
                won = 1
            else:
                won_caption = "It's a tie!"
                won = 3
            if not self.training:
                # set the display caption
                pygame.display.set_caption(won_caption)
                # update the players screen
                pygame.display.flip()

        return won
        
    def reset(self):
        self.grid_status = np.zeros(self.grid_size*self.grid_size, np.int)
        self.edge_status = np.zeros(2*self.grid_size*(self.grid_size+1), np.dtype(bool))
        self.h_boxes = 0
        self.c_boxes = 0
        #self.grid_status = np.array([1,2,0,2])
        #self.edge_status = np.array([False,False,True,True,False,False,True,False,False,True,False,False])
        
        if not self.training:
            pygame.init()
            self.screen=pygame.display.set_mode([self.GRIDPIXEL*self.grid_size+2*self.BOUNDARYPIXEL+self.DOTWIDTH, 
                                                   self.GRIDPIXEL*self.grid_size+2*self.BOUNDARYPIXEL+self.DOTWIDTH])
            pygame.display.set_caption('Dots and Boxes')
            
            self.caption = "'s turn    "
            
            # load all images
            self.empty = pygame.image.load("pics/empty.png")
            self.H = pygame.image.load("pics/H.png")
            self.C = pygame.image.load("pics/C.png")
            self.block = pygame.image.load("pics/block.png")
            self.lineX = pygame.image.load("pics/lineX.png")
            self.lineXempty = pygame.image.load("pics/lineXempty.png")
            self.lineY = pygame.image.load("pics/lineY.png")
            self.lineYempty = pygame.image.load("pics/lineYempty.png")
            
            self.show()
    
    def getedgechoices(self):
        return [index for index in range(len(self.edge_status)) if self.edge_status[index]==False]
    
    def executemove(self,move):
        reward = np.array([0,0]) #[C,H]
        
        self.edge_status[move] = True
        if not self.training:   
            self.show()
                   
        number = self.checkclosedbox(move)
                    
        if number == 0:
            if self.turn == "H":
                self.turn = "C"
            elif self.turn=="C":
                self.turn = "H"
        else :
            if self.turn == "H":
                reward += np.array([-number,number])
            elif self.turn=="C":
                reward += np.array([number,-number])
        if not self.training:   
            self.show()
                    
        won = self.won()
        if won>0: 
            if not self.training:
                self.accept_clicks = False
            if won == 1: 
                #H
                reward += np.array([-number*5,number*5])
            elif won==2 :
                #C
                reward += np.array([number*5,-number*5])
        return [reward,won]
        
    def saveQfiles(self,iterations):
        Cstr = "temp/C_"+str(self.grid_size)+"_"+str(iterations)+"_Qtable"
        #Hstr = "H_"+str(self.grid_size)+"_"+str(iterations)+"_Qtable"
        nfiles = self.C1.saveQtable(Cstr)
        return nfiles
        #self.C2.saveQtable(Hstr)
                        
    
    def train(self,iterations,Cplayer1):
        self.training = True
        self.C1 = Cplayer1#C
        
        for i in range(iterations):
            self.reset()
            self.C1.reset() 
            #self.C2.reset() 
            
            won = self.won()
            self.turn = random.choice(["H", "C"])
            Tstart=time.time()
            while won==0:
                possiblemoves = self.getedgechoices()
                Q_state = np.concatenate((self.edge_status,self.grid_status))
                
                if self.turn == "C":
                    move = self.C1.choosemove(Q_state,possiblemoves)
                else:
                    move = random.choice(possiblemoves)
                    
                [reward,won] = self.executemove(move)
                
                possiblemoves = self.getedgechoices()
                Q_state = np.concatenate((self.edge_status,self.grid_status))
                self.C1.updateQ(reward[0],Q_state,possiblemoves)
            Tend =  time.time()
            Ielapsed = Tend-Tstart
            print("training ",self.grid_size,i,"\'s epoch; time ",str(Ielapsed))
        nfiles = self.saveQfiles(iterations)
        return nfiles
        
    
    
    def show(self):
        self.screen.fill(0)
        
        for col in range(self.grid_size+1):
            for row in range(self.grid_size+1):
                
                x = col * self.GRIDPIXEL+self.BOUNDARYPIXEL
                y = row * self.GRIDPIXEL+self.BOUNDARYPIXEL
                
                if col<self.grid_size and row<self.grid_size:
                    if self.grid_status[row*self.grid_size+col] == 0:
                        self.screen.blit(self.empty, (x, y))
                    elif self.grid_status[row*self.grid_size+col] == -1:
                        self.screen.blit(self.H, (x, y))
                    elif self.grid_status[row*self.grid_size+col] == 1:
                        self.screen.blit(self.C, (x, y))
                
                if col>0 and row>0:
                    x -= self.GRIDPIXEL
                    if not self.edge_status[(2*self.grid_size+1)*row+col-1]:
                        self.screen.blit(self.lineXempty, (x, y))
                    else:
                        self.screen.blit(self.lineX, (x, y))
                    x += self.GRIDPIXEL
                    y -= self.GRIDPIXEL
                    if not self.edge_status[self.grid_size+(2*self.grid_size+1)*(row-1)+col]:
                        self.screen.blit(self.lineYempty, (x, y))
                    else:
                        self.screen.blit(self.lineY, (x, y))
                elif col==0 and row>0:  
                    y -= self.GRIDPIXEL
                    if not self.edge_status[self.grid_size+(2*self.grid_size+1)*(row-1)+col]:
                        self.screen.blit(self.lineYempty, (x, y))
                    else:
                        self.screen.blit(self.lineY, (x, y))
                elif row==0 and col>0:
                    x -= self.GRIDPIXEL
                    if not self.edge_status[(2*self.grid_size+1)*row+col-1]:
                        self.screen.blit(self.lineXempty, (x, y))
                    else:
                        self.screen.blit(self.lineX, (x, y))
                        
        for col in range(self.grid_size+1):
            for row in range(self.grid_size+1):                
                x = col * self.GRIDPIXEL+self.BOUNDARYPIXEL
                y = row * self.GRIDPIXEL+self.BOUNDARYPIXEL
                self.screen.blit(self.block, (x, y))

        pygame.display.set_caption(self.turn + self.caption + "     H:" + str(self.h_boxes) + "   C:" + str(
            self.c_boxes))
        pygame.display.flip()
    
    
    
    def checkclick(self,x,y):
        index = 0;
        if (x<self.BOUNDARYPIXEL or x>self.GRIDPIXEL*self.grid_size+self.BOUNDARYPIXEL+self.DOTWIDTH or
           y<self.BOUNDARYPIXEL or y>self.GRIDPIXEL*self.grid_size+self.BOUNDARYPIXEL+self.DOTWIDTH):
            return [False, index]
        x -= self.BOUNDARYPIXEL
        y -= self.BOUNDARYPIXEL
        
        ax = x // self.GRIDPIXEL
        bx = x % self.GRIDPIXEL
        ay = y // self.GRIDPIXEL
        by = y % self.GRIDPIXEL
        
        if ax >self.grid_size and bx < self.DOTWIDTH:
            if by > self.DOTWIDTH:
                index = ay*(2*self.grid_size+1)+ax+self.grid_size
                if not self.edge_status[index]:
                    return [True, index]
                
        elif ay >self.grid_size and by < self.DOTWIDTH:
            if bx > self.DOTWIDTH:
                index = ay*(2*self.grid_size+1)+ax
                if not self.edge_status[index]:
                    return [True, index]
            
        else:        
            if bx<20 and by>20:
                index = ay*(2*self.grid_size+1)+ax+self.grid_size
                if not self.edge_status[index]:
                    return [True, index]
            elif bx>20 and by<20:
                index = ay*(2*self.grid_size+1)+ax
                if not self.edge_status[index]:
                    return [True, index]
            else:
                return [False, index]
            
        return [False,index]
    
    def randomplay(self,trainingiterations,playtimes,Cplayer,nfiles):
        Performance = [0,0,0]
        self.C3 = Cplayer
        Cstr = "temp/C_"+str(self.grid_size)+"_"+str(trainingiterations)+"_Qtable"
        self.C3.loadQtable(Cstr,nfiles)
        self.C3.epsilon = 0
        self.training = True
        for i in range(playtimes):
            self.turn = random.choice(["H","C"])
            self.reset()
            self.C3.reset()
            
            won = 0
            while won==0:
                if self.turn =="H":
                    possiblemoves = self.getedgechoices()
                    move = random.choice(possiblemoves)
                    [reward,won] = self.executemove(move)
                else:
                    possiblemoves = self.getedgechoices()
                    Q_state = np.concatenate((self.edge_status,self.grid_status))
                    move = self.C3.choosemove(Q_state,possiblemoves)
                    [reward,won] = self.executemove(move)
                
                if won == 1:
                    Performance[0] += 1
                elif won == 2:
                    Performance[1] += 1
                elif won ==3:
                    Performance[2] += 1
            print(Performance)
        return Performance
                    
        
    def play(self,trainingiterations,Cplayer,nfiles):
        self.training = False
        self.reset()
        
        self.C3 = Cplayer#C
        Cstr = "temp/C_"+str(self.grid_size)+"_"+str(trainingiterations)+"_Qtable"
        self.C3.loadQtable(Cstr,nfiles)
        self.C3.epsilon = 0
        self.turn = "H"
        
        won = 0
        while won==0:
            if self.turn == "H":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit(0)
                    elif event.type == pygame.MOUSEBUTTONDOWN and pygame.mouse.get_pressed()[0]:
                        if not self.accept_clicks:
                            continue
                        
                        # get the current position of the cursor
                        x = pygame.mouse.get_pos()[0]
                        y = pygame.mouse.get_pos()[1]
                        #print(x)
                        #print(y)
                        
                        [onplace, edgeindex] = self.checkclick(x,y)
                        #print([onplace, edgeindex])
                        
                        if onplace:
                            [reward,won] = self.executemove(edgeindex)
            else :
                possiblemoves = self.getedgechoices()
                Q_state = np.concatenate((self.edge_status,self.grid_status))
                move = self.C3.choosemove(Q_state,possiblemoves)
                [reward,won] = self.executemove(move)
                    #print(self.grid_status)
                    #print(self.edge_status)
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit(0)
                    
    def trainNN(self,iterations,Cplayer1):
        self.training = True
        self.C1 = Cplayer1#C
        #self.C2 = Cplayer2#H
        Tstart = time.time()
        for i in range(iterations):
            
            self.turn = random.choice(["H", "C"])
            
            self.reset()
            self.C1.reset() 
            
            won = self.won()
            
            while won==0:
                possiblemoves = self.getedgechoices()
                Q_state = np.concatenate((self.edge_status,self.grid_status))
                
                if self.turn == "C":
                    move = self.C1.choosemove(Q_state,possiblemoves)
                else:
                    move = random.choice(possiblemoves)
                    
                [reward,won] = self.executemove(move)
                
                possiblemoves = self.getedgechoices()
                Q_state = np.concatenate((self.edge_status,self.grid_status))
                
                #print(self.turn)
                #print("C1: ")
                self.C1.updateNN(reward[0],Q_state,possiblemoves,won)
            
            Ielapsed = time.time()-Tstart
            Tstart = time.time()
            print("training ",self.grid_size,i,"\'s epoch; time ",str(Ielapsed),"cost: ",self.C1.cost.history['loss'][0])
        self.saveQfiles(iterations)