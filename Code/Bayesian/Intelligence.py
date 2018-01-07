# coding=UTF8

# Python TicTacToe game with Tk GUI and minimax AI
# Author: Maurits van der Schee <maurits@vdschee.nl>

# This is a implementation of a bayesian intelligence. The agent assumes that only the present states values matter and all
# previous states can be ignored. Due to the agent being bayesian it models the world in a bayesian manner.
# Thus the intelligence cannot overfit and will always arrive at the optimal solution given infinite games against a random agent.
import sys
import numpy as np
import random
#import copy
if sys.version_info >= (3, 0):

  from tkinter import Tk, Button
  from tkinter.font import Font
else:
  from Tkinter import Tk, Button
  from tkFont import Font
from copy import deepcopy

class Board:
  def __init__(self,other=None):
    self.learn = True

    self.states = []
    self.table = np.ones((3,3,3,3,3,3,3,3,3,3))

    self.empty = 0
    self.player = 1
    self.opponent = 2


    self.convert = {}

    self.convert[self.empty] = ' '
    self.convert[self.player] = 'X'
    self.convert[self.opponent] = 'O'

    self.size = 3
    self.fields = np.zeros((self.size,self.size),dtype=int)
    for y in range(self.size):
      for x in range(self.size):
        self.fields[x][y] = self.empty
    # copy constructor
    if other:
      self.__dict__ = deepcopy(other.__dict__)

  def reset(self):
    for y in range(self.size):
      for x in range(self.size):
        self.fields[x][y] = self.empty
    self.player = 1
    self.opponent = 2
    self.empty = 0
  def move(self,x,y):
    board = Board(self)
    board.fields[y][x] = board.player
    (board.player,board.opponent) = (board.opponent,board.player)
    return board

  def saveState(self,saveState):
    self.states.append(saveState)


  def learnGame(self,score):
    print("Score: ",score)
    #print("Number of states: ", len(self.states), "Score: ",score)
    #print("States:\n")
    for curState in self.states:
      print(curState,"\n")
      self.table[score][curState[0]][curState[1]][curState[2]][curState[3]][curState[4]][curState[5]][curState[6]][curState[7]][curState[8]]+=1
      print("Here",self.table[score][curState[0]][curState[1]][curState[2]][curState[3]][curState[4]][curState[5]][curState[6]][curState[7]][curState[8]])
    self.save()
    self.states = []

  def save(self):
    return np.save("intelMemory.npy", self.table)

  def load(self):
    self.table = np.load("intelMemory.npy")

  def intelMove(self):
    posValues = []
    moveList = []
    for i in range(3):
      for j in range(3):
        #print(self.fields[i,j])
        posValues.append(self.fields[i][j])
        if self.fields[i][j]==self.empty:
          moveList.append(i*3+j)
    print("Pos: ", posValues)
    print("MoveList: ",moveList)
    moveProbs = np.zeros(len(moveList))
    sumProbs = 0.0
    for i in range(len(moveList)):
      temp = posValues.copy()
      temp[moveList[i]] = 2

      moveProbs[i] = self.table[1][temp[0]][temp[1]][temp[2]][temp[3]][temp[4]][temp[5]][temp[6]][temp[7]][temp[8]]# + self.table[2][temp[0]][temp[1]][temp[2]][temp[3]][temp[4]][temp[5]][temp[6]][temp[7]][temp[8]]
      totalProb = moveProbs[i] + self.table[0][temp[0]][temp[1]][temp[2]][temp[3]][temp[4]][temp[5]][temp[6]][temp[7]][temp[8]]
      moveProbs[i]/=totalProb
      print("Test: ",temp,". Prob: ",moveProbs[i])
      sumProbs += moveProbs[i]
    #print("Move prob: ",moveProbs)
    #print("Sum probs: ",sumProbs)
    randomValue = random.random()*sumProbs
    #print("Random value: ",randomValue)

    for i in range(len(moveProbs)):
      randomValue -= moveProbs[i]
      if randomValue>0.0:
        break
    i = np.argmax(moveProbs) # Take out
    print("i: ",i)
    return (int(moveList[i]%3),int(moveList[i]/3))

  def __minimax(self, player):
    if self.won():
      if player:
        return (-1,None)
      else:
        return (+1,None)
    elif self.tied():
      return (0,None)
    elif player:
      best = (-2,None)
      for x,y in self.fields:
        if self.fields[x,y]==self.empty:
          value = self.move(x,y).__minimax(not player)[0]
          if value>best[0]:
            best = (value,(x,y))
      return best
    else:
      best = (+2,None)
      for x,y in self.fields:
        if self.fields[x,y]==self.empty:
          value = self.move(x,y).__minimax(not player)[0]
          if value<best[0]:
            best = (value,(x,y))
      return best
  
  def best(self):
    if self.learn:
      move = self.intelMove()
      return move
    else:
      move = self.__minimax(True)[1]
      #print(move)
      return move
  
  def tied(self):
    for y in range(len(self.fields)):
      for x in range(len(self.fields[0])):
        if self.fields[y][x]==self.empty:
          return False
    return True
  
  def won(self):
    # horizontal
    for y in range(self.size):
      winning = []
      for x in range(self.size):
        if self.fields[x,y] == self.opponent:
          winning.append((x,y))
      if len(winning) == self.size:
        return winning
    # vertical
    for x in range(self.size):
      winning = []
      for y in range(self.size):
        if self.fields[x,y] == self.opponent:
          winning.append((x,y))
      if len(winning) == self.size:
        return winning
    # diagonal
    winning = []
    for y in range(self.size):
      x = y
      if self.fields[x,y] == self.opponent:
        winning.append((x,y))
    if len(winning) == self.size:
      return winning
    # other diagonal
    winning = []
    for y in range(self.size):
      x = self.size-1-y
      if self.fields[x,y] == self.opponent:
        winning.append((x,y))
    if len(winning) == self.size:
      return winning
    # default
    return None
  
  def __str__(self):
    string = ''
    for y in range(self.size):
      for x in range(self.size):
        string+=self.fields[x,y]
      string+="\n"
    return string
        
class GUI:
  def __init__(self):
    self.app = Tk()
    self.app.title('TicTacToe')
    self.app.resizable(width=False, height=False)
    self.board = Board()
    self.board.load() # Load previously learnt values
    self.font = Font(family="Helvetica", size=32)
    self.buttons = []
    for y in range(len(self.board.fields)):
      for x in range(len(self.board.fields[0])):
        #print(x," ",y)
        handler = lambda x=x, y=y: self.move(x,y)
        #handler = lambda: self.reset()
        button = Button(self.app, command=handler, font=self.font, width=2, height=1)
        button.grid(row=y, column=x)
        self.buttons.append(button)
    handler = lambda: self.reset()
    button = Button(self.app, text='reset', command=handler)
    button.grid(row=self.board.size+1, column=0, columnspan=self.board.size, sticky="WE")
    '''handler = lambda: self.move(0, 0)
    button = Button(self.app, command=handler, font=self.font, width=2, height=1)
    button.grid(row=y, column=x+1)
    self.update(0)'''

  def reset(self):
    self.board.reset()
    self.update(0)
  
  def move(self,x,y):
    #print("Move: ",x," ",y)
    #print("Move: ",x," ",y)
    #print("Moving!!")
    self.app.config(cursor="watch")
    self.app.update()
    self.board = self.board.move(x,y)
    end = self.update(0)
    if not end:
      #print("End: ",end)
      move = self.board.best()
      if move:
        self.board = self.board.move(*move)

        curState = []

        for i in range(3):
          for j in range(3):
            entry = self.board.fields[i][j]
            curState.append(entry)
        #print(curState)
        self.board.saveState(curState)
        self.update(1)
    self.app.config(cursor="")
            
  def update(self,player):
    end = False
    for x in range(len(self.board.fields)):
      for y in range(len(self.board.fields[0])):
        text = self.board.convert[self.board.fields[x][y]]
        self.buttons[x*3+y]['text'] = text
        self.buttons[x*3+y]['disabledforeground'] = 'black'
        if self.board.fields[x][y]==self.board.empty:
          self.buttons[x*3+y]['state'] = 'normal'
        else:
          #print("Disabled1!")
          self.buttons[x*3+y]['state'] = 'disabled'
    winning = self.board.won()
    if winning:
      end = True
      #print("End Game1")
      self.board.learnGame(player*2)

      for x,y in winning:
        self.buttons[x*3+y]['disabledforeground'] = 'red'
      for x in range(len(self.board.fields)):
        for y in range(len(self.board.fields[0])):
          #print("Disabled2!")
          self.buttons[x*3+y]['state'] = 'disabled'
    elif self.board.tied():
      #print("End Game2")
      self.board.learnGame(1)
      end = True
    for x in range(len(self.board.fields)):
      for y in range(len(self.board.fields[0])):
        self.buttons[x*3+y].update()
    return end

  def mainloop(self):
    self.app.mainloop()

if __name__ == '__main__':
  GUI().mainloop()
