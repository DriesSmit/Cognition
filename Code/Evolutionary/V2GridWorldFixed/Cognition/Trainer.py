from __future__ import division
import matplotlib.pyplot as plt
from gridworld import gameEnv

env = gameEnv(partial=False,size=3)
s = env.reset()

for i in range(1000):
    arr = env.renderEnv()
    plt.imshow(arr)
    plt.pause(0.0001)
    s1, r, d = env.step(i%4)

import Cognition
import numpy as np
import random
#from playsound import playsound
#np.random.seed(5)

def bubbleSort(reward,weights):
    for passnum in range(len(reward)-1,0,-1):
        for i in range(passnum):
            if reward[i]<reward[i+1]:
                temp = reward[i]
                reward[i] = reward[i+1]
                reward[i+1] = temp

                tempW = list(weights[i])
                weights[i] = list(weights[i + 1])
                weights[i+1] = tempW

                #print(weights[i][0],weights[i+1][0])

def breed(w1, w2,rate):
  wFinal = np.zeros(len(w1))
  for i in range(len(w1)):
    val = random.uniform(0.0,rate)
    wFinal[i] = (w1[i]*val + w2[i]*(1.0-val))
  return wFinal

# the function we want to optimize
def testIntel(w):
    pass

def outputIntel(w):
    pass
