import numpy as np
import math
class Cognition(object):
  def __init__(self,w,inoutSize):
    self.w = w
    if l<2:
      l=2
    if d<2:
      d=2

    if inoutSize[0]<1:
      inoutSize[0]=1
    if inoutSize[1]<1:
      inoutSize[1]=1

    self.l = l
    self.d = d

    self.input = np.zeros(inoutSize[0])
    self.output = np.zeros(inoutSize[1])
    self.network = []

    wCounter = 0
    layer = np.zeros((l,l,inoutSize[0]+8))  #1 internal value + 1 self weight + 1 bias + 5 external weights + input weights
    for i in range(self.l):
      for j in range(self.l):
        for k in range(inoutSize[0]+8):
          layer[i][j][k] = w[wCounter]
          wCounter += 1

    self.network.append(layer)
    for i1 in range(d-2):
      layer = np.zeros((l,l, 9)) #1 internal value + 1 self weight + 1 bias + 6 external weights + input weights
      for i2 in range(self.l):
        for i3 in range(self.l):
          for i4 in range(9):
            layer[i2][i3][i4] = w[wCounter]
            wCounter += 1
      self.network.append(layer)

    layer = np.zeros((l, l, 8+inoutSize[1])) #1 internal value + 1 preLayer_weight + input weights + 4 external weights + 1 self weight + 1 bias
    for i in range(self.l):
      for j in range(self.l):
        for k in range(8+inoutSize[1]):
          layer[i][j][k] = w[wCounter]
          wCounter += 1
    self.network.append(layer)

  def setInput(self,i):
    self.input = i

  def getOutput(self):
    #print(self.output)
    return self.output#sigmoid(self.w[0]+self.w[1]-self.w[4])*2.0*self.input[0]-sigmoid(self.w[1] + self.w[2])*2.0*self.input[1]

    #self.output
    #sigmoid(self.w[0])*2.0*self.input[0]-sigmoid(self.w[1])*2.0*self.input[1]
    #1.0*self.input[0]+0.0*self.input[1]

  def nextState(self):
    tempLayer = np.zeros((self.l,self.l))
    for i in range(self.l):
      for j in range(self.l):
        tempLayer[i][j] += self.network[0][i    ][j    ][0] * self.network[0][i][j][1]
        tempLayer[i][j] += self.network[0][i    ][j    ][2]
        tempLayer[i][j] += self.network[0][i    ][(j + 1)%self.l][0] * self.network[0][i][j][3]
        tempLayer[i][j] += self.network[0][i    ][j - 1][0] * self.network[0][i][j][4]
        tempLayer[i][j] += self.network[0][(i + 1)%self.l][j    ][0] * self.network[0][i][j][5]
        tempLayer[i][j] += self.network[0][i - 1][j    ][0] * self.network[0][i][j][6]
        tempLayer[i][j] += self.network[1][i    ][j    ][0] * self.network[0][i][j][7]
        for k in range(len(self.input)):
          tempLayer[i][j] += self.input[k]*self.network[0][i][j][k+8]
        tempLayer[i][j] = sigmoid(tempLayer[i][j])

    for d in range(1,self.d-1):
      for i in range(self.l):
        for j in range(self.l):
          tempValue = 0.0
          tempValue += self.network[d - 1][i    ][j    ][0] * self.network[d][i][j][1] # Previous layer neuron
          tempValue += self.network[d    ][i    ][j    ][0] * self.network[d][i][j][2]
          tempValue += self.network[d    ][i    ][j    ][3]
          tempValue += self.network[d    ][i    ][(j + 1)%self.l][0] * self.network[d][i][j][4]
          tempValue += self.network[d    ][i    ][j - 1][0] * self.network[d][i][j][5]
          tempValue += self.network[d    ][(i + 1)%self.l][j    ][0] * self.network[d][i][j][6]
          tempValue += self.network[d    ][i - 1][j    ][0] * self.network[d][i][j][7]
          tempValue += self.network[d + 1][i    ][j    ][0] * self.network[d][i][j][8]
          tempValue = sigmoid(tempValue)

          self.network[d - 1][i][j][0] = tempLayer[i][j]
          tempLayer[i][j] = tempValue

    for i in range(self.l):
      for j in range(self.l):
        tempValue = 0.0
        tempValue += self.network[self.d - 2][i    ][j    ][0] * self.network[self.d - 1][i][j][1]
        tempValue += self.network[self.d - 1][i    ][j    ][0] * self.network[self.d - 1][i][j][2]
        tempValue += self.network[self.d - 1][i    ][j    ][3]
        tempValue += self.network[self.d - 1][i    ][(j + 1)%self.l][0] * self.network[self.d - 1][i][j][4]
        tempValue += self.network[self.d - 1][i    ][j - 1][0] * self.network[self.d - 1][i][j][5]
        tempValue += self.network[self.d - 1][(i + 1)%self.l][j    ][0] * self.network[self.d - 1][i][j][6]
        tempValue += self.network[self.d - 1][i - 1][j    ][0] * self.network[self.d - 1][i][j][7]
        tempValue = sigmoid(tempValue)

        self.network[self.d - 2][i][j][0] = tempLayer[i][j]
        self.network[self.d - 1][i][j][0] = tempValue

    for i in range(len(self.output)):
      self.output[i] = 0.0
      for j in range(self.l):
        for k in range(self.l):
          self.output[i] += self.network[self.d-1][j][k][0] * self.network[self.d-1][j][k][i+8]
          #print(self.network[self.d-1][j][k][0], " ", self.network[self.d-1][j][k][i+8])
      self.output[i] = sigmoid(self.output[i])
def getNumWeight(io,l,d):
  return l*l*((io[0]+8) + (d-2)*9 + (8+io[1]))+1

def sigmoid(input):
  if input >-100:
    return 1.0/(1-math.pow(math.e,-input))
  else:
    return 1

  '''if(input>0.5):
    return 1.0
  else:
    return 0.0'''

def save(w,fileName):
  f1 = open(fileName, 'w+')
  for i in range(len(w)):
    f1.write(str(w[i])+"\n")
  f1.close()

def load(fileName):
    with open(fileName, 'r') as f:
      content = f.readlines()
    content = [x.strip() for x in content]
    f.close()
    return content