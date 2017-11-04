import random
import numpy as np
class Game(object):
    def __init__(self):
        self.s = [-1, random.getrandbits(1)]
        #self.s = random.getrandbits(1)
    def getState(self):
        arr = np.zeros(1)
        arr[0] = self.s[1]
        return arr
    def getFullState(self):
        return self.s
    def nextState(self,move):
        rGame = 0
        if self.s[0]>-0.5:
            r = 1.0-abs(float(move)-float(self.s[0]))
        else:
            r = 1.0
            rGame = 1
        if abs(float(move) - float(self.s[0])) < 0.5:
            rGame = 1
        self.s[0] = self.s[1]
        self.s[1] = random.getrandbits(1)

        arr = np.zeros(1)
        arr[0] = self.s[1]
        return r,arr,rGame

def inoutSize():
    inout = [2, 1]
    return inout
