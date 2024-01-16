import numpy as np
import math 

trueW = np.array([1, 2, 3, 4, 5])

def generate():
  x = np.random.randn(len(trueW))
  y = trueW.dot(x) + np.random.randn()
  #print('example', x, y)
  return (x, y)
  
trainEx = [generate() for i in range(1000000)]

def phi(x):
  return np.array(x)

def initialWeightVector():
  return np.zeros(len(trueW))

def trainLoss(w):
  return 1.0/len(trainEx) * sum((w.dot(phi(x)) - y)**2 for x, y in trainEx)

def gradientTrainLoss(w):
  return 1.0/len(trainEx) * sum(2*(w.dot(phi(x)) - y)*phi(x) for x, y in trainEx)

def loss(w, i):
  x, y = trainEx[i]
  return (w.dot(phi(x)) - y)**2

def gradientLoss(w, i):
  x, y = trainEx[i]
  return 2 * (w.dot(phi(x)) - y) * phi(x)

############################################################################
# Optimization Algorithm
def stochasticGradientDescent(f, gradientF, initialWeightVector):
  w = initialWeightVector()
  numUpdates = 0
  for t in range(500):
    for i in range(len(trainEx)):
        value = f(w, i)
        gradient = gradientF(w, i)
        numUpdates += 1
        eta = 1.0 / math.sqrt(numUpdates)
        w = w - eta * gradient
    print(f"epoch {t}: w= {w}, F(w) = {value}, gradientF = {gradient}")

stochasticGradientDescent(loss, gradientLoss, initialWeightVector)