import numpy as np

trainEx = [(1, 1), (2, 3), (4, 3)]

def phi(x):
  return np.array([1, x])

def initialWeightVector():
  return np.zeros(2)

def trainLoss(w):
  return 1.0/len(trainEx) * sum((w.dot(phi(x)) - y)**2 for x, y in trainEx)

def gradientTrainLoss(w):
  return 1.0/len(trainEx) * sum(2*(w.dot(phi(x)) - y)*phi(x) for x, y in trainEx)

############################################################################
# Optimization Algorithm
def gradientDescent(F, gradientF, initialWeightVector):
  w = initialWeightVector()
  eta = 0.1
  for t in range(500):
    value = F(w)
    gradient = gradientF(w)
    w = w - eta * gradient
    print(f"epoch {t}: w= {w}, F(w) = {value}, gradientF = {gradient}")

gradientDescent(trainLoss, gradientTrainLoss, initialWeightVector)