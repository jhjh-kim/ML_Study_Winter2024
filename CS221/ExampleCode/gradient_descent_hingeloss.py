import numpy as np

#(x, y) pairs
trainEx = [((0, 2), 1), ((-2, 0), 1), ((1, -1), -1)]

def phi(x):
  return np.array(x)

def initialWeightVector():
  return np.zeros(2)

def trainLoss(w):
  return 1.0/len(trainEx) * sum(max(1 - w.dot(phi(x))*y, 0) for x, y in trainEx)

def gradientTrainLoss(w):
  return 1.0/len(trainEx) * sum(-(phi(x)*y) if 1 - w.dot(phi(x))*y > 0 else 0 for x, y in trainEx)

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