"""
A bare bones examples of optimizing a black-box function (f) using
Natural Evolution Strategies (NES), where the parameter distribution is a 
gaussian of fixed standard deviation.
"""
import Game
import Cognition
import numpy as np
import random
from playsound import playsound
#np.random.seed(5)

# the function we want to optimize
def testIntel(w):
  reward = 0.0
  rGameTotal = 0
  g = Game.Game()
  c = Cognition.Cognition(w,l,d,Game.inoutSize())
  #print("Game start")
  s = g.getState()
  for i in range(roundsPerGame):
    arr = np.zeros(2)
    arr[0] = s[0]
    arr[1] = 1
    c.setInput(arr)
    c.nextState()
    arr[0] = s[0]
    arr[1] = 0
    c.setInput(arr)
    for j in range(statesPerMove-1):
      c.nextState()
    #preState = g.getFullState()

    move = c.getOutput()*2.0-0.5

    if move < 0.0:
      move = 0.0
    elif move>1.0:
      move = 1.0

    r, s, rGame = g.nextState(move)
    #print(r)
    #print("Input to intel: ", preState, ". Output of intel: " ,move, "Reward: ",r)
    rGameTotal += rGame
    reward += r
  #print("Game ends")
  return reward,float(rGameTotal/roundsPerGame)

def outputIntel(w):
  reward = 0.0
  rGameTotal = 0
  g = Game.Game()
  c = Cognition.Cognition(w,l,d,Game.inoutSize())
  #print("Game start")
  s = g.getState()
  for i in range(roundsPerGame):
    arr = np.zeros(2)
    arr[0] = s[0]
    arr[1] = 1
    print("Input1: ",arr)
    c.setInput(arr)
    c.nextState()
    arr[0] = s[0]
    arr[1] = 0
    print("Input2: ", arr)
    c.setInput(arr)
    for j in range(statesPerMove-1):
      c.nextState()
    #preState = g.getFullState()

    move = c.getOutput()*2.0-0.5

    if move < 0.0:
      move = 0.0
    elif move>1.0:
      move = 1.0

    r, s, rGame = g.nextState(move)
    print("Intel move: ", move, "Reward: ",r,". Game reward: ",rGame)
    rGameTotal += rGame
    reward += r
  #print("Game ends")
  return reward,float(rGameTotal/roundsPerGame)

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

# hyperparameters
npop = 20 # population size
#sigma = 0.01 # noise standard deviation
alpha = 1.0 # learning rate
roundsPerGame = 30
#Intelligence
statesPerMove = 10
l = 3
d = 3
saveNum = 3
loadIntel = True
trainIntel = False
# start the optimization
io = Game.inoutSize()
numWeights = Cognition.getNumWeight(io,l,d)
print('Number of weights: ',numWeights)
w = np.random.randn(npop,numWeights) # our initial guess is random

for i in range(npop):
  w[i][0] = random.uniform(0.0,1.0)

if(loadIntel==True):
  for j in range(saveNum):
    wLoad = Cognition.load("Intelligence"+str(j+1)+".txt")

    temp = np.array(wLoad)
    w[j] = temp.astype(np.float)
if trainIntel:
  maxPercentage = 0
  for i in range(1000000):
    # print current fitness of the most likely parameter setting
    #if i % 20 == 0:
    res = testIntel(w[0][1:])
    #print('iter ',i,'.Continoues Reward: ',res[0]/(roundsPerGame),'. Actual percentage: ',round(res[1]*100.0,2),'%')
    print('iter ', i, '.Continoues Reward: ', round(res[0] / (roundsPerGame),2), '. Actual percentage: ', round(res[1] * 100.0, 2),
          '%')

    if i%20==0:
      print("Saving top ",saveNum," intelligences.")
      for j in range(saveNum):
        Cognition.save(w[j],"Intelligence"+str(j+1)+".txt")

    if(res[1]*100.0>=maxPercentage):
      maxPercentage = res[1]*100.0
      playsound('./pass.wav', block=False)

    R = np.zeros(npop)
    rTot = 0.0

    #Test intelligence
    for j in range(npop):
      R[j], _ = testIntel(w[j][1:])  # evaluate the jittered version
      rTot += R[j]
    bubbleSort(R,w)
    rNor = R / rTot
    #print(R)

    #Evolve inteligence
    N = np.random.randn(npop, len(w[0]) - 1)  # samples from a normal distribution N(0,1)
    for j in range(5,npop):
      val = random.uniform(0.0, 0.9)
      index1 = -1

      while (val > 0.0):
        index1 += 1
        val -= rNor[index1]
      val = random.uniform(0.0, 0.9)
      index2 = -1
      while (val > 0.0):
        index2 += 1
        val -= rNor[index2]

      w[j] = breed(w[index1], w[index2],0.0)

      w[j][0] += random.normalvariate(0.0, 0.1*w[j][0])
      w[j][1:] += w[j][0] * N[j]
else:
  res = outputIntel(w[0][1:])
  print('iter ', i, '.Continoues Reward: ', round(res[0] / (roundsPerGame), 2), '. Actual percentage: ',
        round(res[1] * 100.0, 2),
        '%')








