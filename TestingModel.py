from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np
import random
import gym
import pygame
from collections import deque

from IPython.display import clear_output
from matplotlib import pyplot as plt
class SnakeEnv(object):
    def __init__(self):
        self.SCREEN_SIZE = 800
        self.vel = None
        self.gridSize = int(self.SCREEN_SIZE/160)
        self.foodPos = None
        self.spawnPos = None
        self.parts = []
        self.screen = pygame.display.set_mode((self.SCREEN_SIZE, self.SCREEN_SIZE), 0, 32)
        self.surface = pygame.Surface(self.screen.get_size()).convert()
        self.score = 0
        self.clock = pygame.time.Clock()
        pygame.init()
    def getBestAction(self):
        horizAction = -1
        vertAction = -1
        if self.parts[0][0]-self.foodPos[0] < 0:
            horizAction = 3
        elif self.parts[0][0]-self.foodPos[0] > 0:
            horizAction = 2
        if self.parts[0][1] - self.foodPos[1] < 0:
            vertAction = 1
        elif self.parts[0][1] - self.foodPos[1] > 0:
            vertAction = 0
        if horizAction == -1:
            return vertAction
        elif vertAction == -1:
            return horizAction
        elif random.random() > 0.5:
            return horizAction
        return vertAction
    def getNewHeadPos(self):
        a = self.parts[0]
        b = self.vel
        x = (a[0]+b[0])#(+self.gridSize) %self.gridSize
        y = (a[1]+b[1])#(+self.gridSize) %self.gridSize
        return (x,y)
    def rect(self,x,y,w,h,c):
        pygame.draw.rect(self.surface, c, pygame.Rect((x,y), (w,h)))
    def up(self):
        self.vel = (0, -1)
    def down(self):
        self.vel = (0, 1)
    def left(self):
        self.vel = (-1, 0)
    def right(self):
        self.vel = (1, 0)
    def getDistToFood(self):
      return abs(self.foodPos[0]-self.parts[0][0]) + abs(self.foodPos[1]-self.parts[0][1])
    def randomizeFoodPos(self):
        while True:
            self.foodPos = (random.randrange(self.gridSize),random.randrange(self.gridSize))
            if not self.foodPos in self.parts:
                return
    def updateBodyPos(self):
        newHeadPos = self.getNewHeadPos()
        if len(self.parts)>1 and newHeadPos == self.parts[1]:
            return 1
        for i in range(len(self.parts)-1,0,-1):
            self.parts[i] = self.parts[i-1]
        self.parts[0] = self.getNewHeadPos()
        return 0
    def drawBody(self):
        for part in self.parts:
            self.rect((part[0]*(self.SCREEN_SIZE/self.gridSize))+5,(part[1]*(self.SCREEN_SIZE/self.gridSize))+5,(self.SCREEN_SIZE/self.gridSize)-10,(self.SCREEN_SIZE/self.gridSize)-10,(0,0,255))
    def drawFood(self):
        self.rect((self.foodPos[0]*(self.SCREEN_SIZE/self.gridSize))+10,(self.foodPos[1]*(self.SCREEN_SIZE/self.gridSize))+10,(self.SCREEN_SIZE/self.gridSize)-20,(self.SCREEN_SIZE/self.gridSize)-20,(255,0,0))
    def reset(self):
        self.parts = [(0,0),(1,0)]
        self.vel = (1,0)
        self.randomizeFoodPos()
        self.score = 0
        return self.getNNInput()
    def getNNInput(self):
        gridInputs = [[0 for i in range(self.gridSize)] for j in range(self.gridSize)]
        for part in self.parts[1:]:
          gridInputs[part[0]][part[1]] = -1
        if self.parts[0][0] >= 0 and self.parts[0][1] >= 0 and self.parts[0][0] < self.gridSize and self.parts[0][1] < self.gridSize:
          gridInputs[self.parts[0][0]][self.parts[0][1]] = -0.5
        gridInputs[self.foodPos[0]][self.foodPos[1]] = 1
        return np.array(gridInputs).flatten()
    def step(self, action):
        frameScore = 0
        if action==0:
            self.up()
        if action==1:
            self.down()
        if action==2:
            self.left()
        if action==3:
            self.right()
        spawnPos = self.parts[-1]
        oldDist = self.getDistToFood()
        err = self.updateBodyPos()
        newDist = self.getDistToFood()
        if err or self.parts[0] in self.parts[1:] or (self.parts[0][0] < 0 or self.parts[0][1] < 0 or self.parts[0][0] >= self.gridSize or self.parts[0][1] >= self.gridSize):
            return self.getNNInput(), -100, True, 'Info'
        elif self.parts[0]==self.foodPos:
            self.parts.append(spawnPos)
            self.score+=1
            if len(self.parts)==self.gridSize*self.gridSize:
              return self.getNNInput(), 10_000, True, 'Info'
            self.randomizeFoodPos()
            frameScore = 1
        return self.getNNInput(), 150*frameScore, False, 'Info'
    def render(self):
        self.surface = pygame.Surface(self.screen.get_size()).convert()
        self.drawBody()
        self.drawFood()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
        self.screen.blit(self.surface, (0,0))
        pygame.display.update()

def playByHand():
    env = SnakeEnv()
    env.reset()
    action = 3
    totalScore = 0
    done = False
    while not done:
        env.render()
        while True:
            out = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP or event.key == pygame.K_w:
                        action = 0
                        out = True
                    if event.key == pygame.K_DOWN or event.key == pygame.K_s:
                        action = 1 
                        out = True
                    if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                        action = 2
                        out = True
                    if event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                        action = 3
                        out = True
            if out:
                break
        daw_, score, done, _ = env.step(action)
        totalScore+=score
        print("Score: "+str(score))
        if done:
            break
    print(totalScore)
def playStepByStep(name):
    env = SnakeEnv()
    model = load_model(name)
    env.reset()
    env.render()
    totalScore = 0
    for i in range(240):
        pred = model.predict(np.expand_dims(env.getNNInput(), axis=0))
        action = np.argmax(pred)
        strAction = ""
        if action==0:
            strAction= "UP"
        elif action==1:
            strAction = "DOWN"
        elif action==2:
            strAction = "LEFT"
        elif action==3:
            strAction = "RIGHT"
        else:
            strAction = "HABD"
        print("I will play "+strAction+" next.")
        while True:
            out = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP or event.key == pygame.K_w:
                        env.randomizeFoodPos()
                    elif event.key == pygame.K_c:
                        out = True
                        break
            if out:
                break
        
        _, score, done, _ = env.step(action)
        env.render()
        totalScore+=score
        print("Score: "+str(score))
        if done:
            break
    print("Total Score: ",str(totalScore))
def playAI(name):
    env = SnakeEnv()
    model = load_model(name)
    env.reset()
    totalScore = 0
    for i in range(240):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.randomizeFoodPos()
        env.clock.tick(8)
        env.render()
        pred = model.predict(np.expand_dims(env.getNNInput(), axis=0))
        action = np.argmax(pred)
        _, score, done, _ = env.step(action)
        totalScore+=score
        print("Score: "+str(score))
        if done:
            break
    print("Total Score: ",str(totalScore))
def writeToFileManual():
    env = SnakeEnv()
    env.reset()
    totalScore = 0
    fileText = ""
    for i in range(240):
        env.render()
        while True:
            out = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP or event.key == pygame.K_w:
                        action = 0
                        out = True
                    if event.key == pygame.K_DOWN or event.key == pygame.K_s:
                        action = 1 
                        out = True
                    if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                        action = 2
                        out = True
                    if event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                        action = 3
                        out = True
            if out:
                break
        daw_, score, done, _ = env.step(action)
        for i in daw_:
            fileText += str(i)+", "
        fileText += str(action)+"\n"
        totalScore+=score
        print("Score: "+str(score))
        if done:
            break
    f = open("outputs.txt", "a")
    f.write(fileText+"End\n")
    f.close()
for i in range(5):
    # playStepByStep('model10000.h5')
    playAI('model10000.h5')
    # playByHand()